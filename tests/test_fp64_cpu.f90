program test_fp64_cpu
  use, intrinsic :: iso_c_binding
  use fortonnx
  implicit none
  type(fortonnx_runtime) :: runtime
  type(fortonnx_options) :: options
  type(fortonnx_status) :: status
  type(fortonnx_tensor) :: inputs(2), outputs(2)
  real(c_double), target :: state(2,3), values(2,3), expected(2,3)
  logical(c_bool), target :: enabled(2,3), valid(2,3)
  real(c_float), target :: wrong(2,3)
  real(c_double), target :: flat_x(256), flat_y(256)
  logical(c_bool), target :: flat_b(256), flat_c(256)
  integer(c_int64_t), parameter :: rank8_shape(8) = 2_c_int64_t
  integer(c_int) :: element_type
  character(len=1024) :: model_path, path
  real(c_double), target :: x1(2), y1(2)
  logical(c_bool), target :: b1(2), c1(2)
  real(c_double), target :: x2(2,2), y2(2,2)
  logical(c_bool), target :: b2(2,2), c2(2,2)
  real(c_double), target :: x3(2,2,2), y3(2,2,2)
  logical(c_bool), target :: b3(2,2,2), c3(2,2,2)
  real(c_double), target :: x4(2,2,2,2), y4(2,2,2,2)
  logical(c_bool), target :: b4(2,2,2,2), c4(2,2,2,2)
  real(c_double), target :: x5(2,2,2,2,2), y5(2,2,2,2,2)
  logical(c_bool), target :: b5(2,2,2,2,2), c5(2,2,2,2,2)
  real(c_double), target :: x6(2,2,2,2,2,2), y6(2,2,2,2,2,2)
  logical(c_bool), target :: b6(2,2,2,2,2,2), c6(2,2,2,2,2,2)
  real(c_double), target :: x7(2,2,2,2,2,2,2), y7(2,2,2,2,2,2,2)
  logical(c_bool), target :: b7(2,2,2,2,2,2,2), c7(2,2,2,2,2,2,2)

  call get_command_argument(1, model_path)
  options%intra_op_threads = 1
  call runtime%initialize(options, status)
  call require_ok()
  call runtime%load('mixed', trim(model_path), status)
  call require_ok()
  call runtime%tensor_type('mixed', fortonnx_input, 1, element_type, status)
  call require_ok()
  if (element_type /= fortonnx_float64) error stop 'wrong input type metadata'
  call runtime%tensor_type('mixed', fortonnx_output, 2, element_type, status)
  call require_ok()
  if (element_type /= fortonnx_bool) error stop 'wrong output type metadata'
  if (c_sizeof(enabled(1,1)) /= 1_c_size_t) error stop 'ONNX bool requires one byte'
  if (transfer(.true._c_bool, 0_c_int8_t) /= 1_c_int8_t) &
      error stop 'ONNX bool requires 0/1 bytes (nvfortran: use -Munixlogical)'
  state = reshape([1.0_c_double + 2.0_c_double**(-40), -1.0_c_double, &
      2.0_c_double**(-44), 2.0_c_double, -2.0_c_double, 0.0_c_double], [2,3])
  enabled = reshape([.true._c_bool, .false._c_bool, .true._c_bool, &
      .false._c_bool, .true._c_bool, .false._c_bool], [2,3])
  expected = state
  where (enabled) expected = state + 2.0_c_double**(-42)
  inputs(1) = fortonnx_host_tensor(enabled, 'enabled')
  inputs(2) = fortonnx_host_tensor(state, 'state')
  outputs(1) = fortonnx_host_tensor(valid, 'valid')
  outputs(2) = fortonnx_host_tensor(values, 'values')
  call runtime%bind('mixed', inputs, outputs, status)
  call require_ok()
  call runtime%run('mixed', status)
  call require_ok()
  if (any(values /= expected)) error stop 'FP64 precision lost'
  if (any(valid .neqv. (enabled .and. state > 0.0_c_double))) error stop 'wrong bool output'
  ! Empty typed views remain invalid; fixing rank handling must not relax this.
  inputs(2) = fortonnx_host_tensor(state(:0,:), 'state')
  call runtime%bind('mixed', inputs, outputs, status)
  if (status%code /= fortonnx_invalid_argument) error stop 'empty FP64 input view accepted'
  inputs(2) = fortonnx_host_tensor(state, 'state')
  outputs(1) = fortonnx_host_tensor(valid(:0,:), 'valid')
  call runtime%bind('mixed', inputs, outputs, status)
  if (status%code /= fortonnx_invalid_argument) error stop 'empty bool output view accepted'
  outputs(1) = fortonnx_host_tensor(valid, 'valid')
  inputs(2) = fortonnx_host_tensor(wrong, 'state')
  call runtime%bind('mixed', inputs, outputs, status)
  call require_mismatch('input', 'float64', 'float32')
  inputs(2) = fortonnx_host_tensor(state, 'state')
  outputs(1) = fortonnx_host_tensor(wrong, 'valid')
  call runtime%bind('mixed', inputs, outputs, status)
  call require_mismatch('output', 'bool', 'float32')
  outputs(1) = fortonnx_pointer_tensor(c_loc(valid), [3_c_int64_t,2_c_int64_t], &
      fortonnx_cpu, 'valid', element_type=fortonnx_bool)
  inputs(2) = fortonnx_pointer_tensor(c_loc(state), [3_c_int64_t,2_c_int64_t], &
      fortonnx_cpu, 'state', element_type=fortonnx_float64)
  call runtime%bind('mixed', inputs, outputs, status)
  call require_ok()
  call runtime%run('mixed', status)
  call require_ok()
  if (any(values /= expected)) error stop 'typed pointer precision lost'
  path = trim(model_path(:len_trim(model_path)-5)) // '_float64_1.onnx'
  call runtime%load('rank', trim(path), status)
  call require_ok()
  x1 = 1.0_c_double + 2.0_c_double**(-40)
  b1 = .true._c_bool
  call runtime%bind('rank', x1, y1, status)
  call require_ok()
  call runtime%run('rank', status)
  call require_ok()
  if (any(y1 /= x1)) error stop 'rank-1 float64 direct bind failed'
  call runtime%bind('rank', fortonnx_host_tensor(x1), fortonnx_host_tensor(y1), status)
  call require_ok()
  call runtime%run('rank', status)
  call require_ok()
  if (any(y1 /= x1)) error stop 'host tensor view failed'
  call runtime%close()
  call runtime%initialize(options, status)
  call require_ok()
  path = trim(model_path(:len_trim(model_path)-5)) // '_bool_1.onnx'
  call runtime%load('rank', trim(path), status)
  call require_ok()
  x1 = 1.0_c_double + 2.0_c_double**(-40)
  b1 = .true._c_bool
  call runtime%bind('rank', b1, c1, status)
  call require_ok()
  call runtime%run('rank', status)
  call require_ok()
  if (any(c1 .neqv. b1)) error stop 'rank-1 bool direct bind failed'
  call runtime%bind('rank', fortonnx_host_tensor(b1), fortonnx_host_tensor(c1), status)
  call require_ok()
  call runtime%run('rank', status)
  call require_ok()
  if (any(c1 .neqv. b1)) error stop 'host tensor view failed'
  call runtime%close()
  call runtime%initialize(options, status)
  call require_ok()
  path = trim(model_path(:len_trim(model_path)-5)) // '_compare_1.onnx'
  call runtime%load('rank', trim(path), status)
  call require_ok()
  x1 = 1.0_c_double + 2.0_c_double**(-40)
  b1 = .true._c_bool
  call runtime%bind('rank', x1, c1, status)
  call require_ok()
  call runtime%run('rank', status)
  call require_ok()
  if (any(.not. c1)) error stop 'rank-1 compare direct bind failed'
  call runtime%bind('rank', fortonnx_host_tensor(x1), fortonnx_host_tensor(c1), status)
  call require_ok()
  call runtime%run('rank', status)
  call require_ok()
  if (any(.not. c1)) error stop 'host tensor view failed'
  call runtime%close()
  call runtime%initialize(options, status)
  call require_ok()
  path = trim(model_path(:len_trim(model_path)-5)) // '_float64_2.onnx'
  call runtime%load('rank', trim(path), status)
  call require_ok()
  x2 = 1.0_c_double + 2.0_c_double**(-40)
  b2 = .true._c_bool
  call runtime%bind('rank', x2, y2, status)
  call require_ok()
  call runtime%run('rank', status)
  call require_ok()
  if (any(y2 /= x2)) error stop 'rank-2 float64 direct bind failed'
  call runtime%bind('rank', c_loc(x2), c_loc(y2), 2_c_int64_t, status, &
      input_type=fortonnx_float64, output_type=fortonnx_float64)
  call require_ok()
  call runtime%run('rank', status)
  call require_ok()
  if (any(y2 /= x2)) error stop 'typed batch pointer failed'
  call runtime%bind('rank', c_loc(x2), c_loc(y2), 2_c_int64_t, status)
  call require_mismatch('input', 'float64', 'float32')
  call runtime%bind('rank', fortonnx_host_tensor(x2), fortonnx_host_tensor(y2), status)
  call require_ok()
  call runtime%run('rank', status)
  call require_ok()
  if (any(y2 /= x2)) error stop 'host tensor view failed'
  call runtime%close()
  call runtime%initialize(options, status)
  call require_ok()
  path = trim(model_path(:len_trim(model_path)-5)) // '_bool_2.onnx'
  call runtime%load('rank', trim(path), status)
  call require_ok()
  x2 = 1.0_c_double + 2.0_c_double**(-40)
  b2 = .true._c_bool
  call runtime%bind('rank', b2, c2, status)
  call require_ok()
  call runtime%run('rank', status)
  call require_ok()
  if (any(c2 .neqv. b2)) error stop 'rank-2 bool direct bind failed'
  call runtime%bind('rank', c_loc(b2), c_loc(c2), 2_c_int64_t, status, &
      input_type=fortonnx_bool, output_type=fortonnx_bool)
  call require_ok()
  call runtime%run('rank', status)
  call require_ok()
  if (any(c2 .neqv. b2)) error stop 'typed batch pointer failed'
  call runtime%bind('rank', c_loc(b2), c_loc(c2), 2_c_int64_t, status)
  call require_mismatch('input', 'bool', 'float32')
  call runtime%bind('rank', fortonnx_host_tensor(b2), fortonnx_host_tensor(c2), status)
  call require_ok()
  call runtime%run('rank', status)
  call require_ok()
  if (any(c2 .neqv. b2)) error stop 'host tensor view failed'
  call runtime%close()
  call runtime%initialize(options, status)
  call require_ok()
  path = trim(model_path(:len_trim(model_path)-5)) // '_compare_2.onnx'
  call runtime%load('rank', trim(path), status)
  call require_ok()
  x2 = 1.0_c_double + 2.0_c_double**(-40)
  b2 = .true._c_bool
  call runtime%bind('rank', x2, c2, status)
  call require_ok()
  call runtime%run('rank', status)
  call require_ok()
  if (any(.not. c2)) error stop 'rank-2 compare direct bind failed'
  call runtime%bind('rank', fortonnx_host_tensor(x2), fortonnx_host_tensor(c2), status)
  call require_ok()
  call runtime%run('rank', status)
  call require_ok()
  if (any(.not. c2)) error stop 'host tensor view failed'
  call runtime%close()
  call runtime%initialize(options, status)
  call require_ok()
  path = trim(model_path(:len_trim(model_path)-5)) // '_float64_3.onnx'
  call runtime%load('rank', trim(path), status)
  call require_ok()
  x3 = 1.0_c_double + 2.0_c_double**(-40)
  b3 = .true._c_bool
  call runtime%bind('rank', x3, y3, status)
  call require_ok()
  call runtime%run('rank', status)
  call require_ok()
  if (any(y3 /= x3)) error stop 'rank-3 float64 direct bind failed'
  call runtime%bind('rank', fortonnx_host_tensor(x3), fortonnx_host_tensor(y3), status)
  call require_ok()
  call runtime%run('rank', status)
  call require_ok()
  if (any(y3 /= x3)) error stop 'host tensor view failed'
  call runtime%close()
  call runtime%initialize(options, status)
  call require_ok()
  path = trim(model_path(:len_trim(model_path)-5)) // '_bool_3.onnx'
  call runtime%load('rank', trim(path), status)
  call require_ok()
  x3 = 1.0_c_double + 2.0_c_double**(-40)
  b3 = .true._c_bool
  call runtime%bind('rank', b3, c3, status)
  call require_ok()
  call runtime%run('rank', status)
  call require_ok()
  if (any(c3 .neqv. b3)) error stop 'rank-3 bool direct bind failed'
  call runtime%bind('rank', fortonnx_host_tensor(b3), fortonnx_host_tensor(c3), status)
  call require_ok()
  call runtime%run('rank', status)
  call require_ok()
  if (any(c3 .neqv. b3)) error stop 'host tensor view failed'
  call runtime%close()
  call runtime%initialize(options, status)
  call require_ok()
  path = trim(model_path(:len_trim(model_path)-5)) // '_compare_3.onnx'
  call runtime%load('rank', trim(path), status)
  call require_ok()
  x3 = 1.0_c_double + 2.0_c_double**(-40)
  b3 = .true._c_bool
  call runtime%bind('rank', x3, c3, status)
  call require_ok()
  call runtime%run('rank', status)
  call require_ok()
  if (any(.not. c3)) error stop 'rank-3 compare direct bind failed'
  call runtime%bind('rank', fortonnx_host_tensor(x3), fortonnx_host_tensor(c3), status)
  call require_ok()
  call runtime%run('rank', status)
  call require_ok()
  if (any(.not. c3)) error stop 'host tensor view failed'
  call runtime%close()
  call runtime%initialize(options, status)
  call require_ok()
  path = trim(model_path(:len_trim(model_path)-5)) // '_float64_4.onnx'
  call runtime%load('rank', trim(path), status)
  call require_ok()
  x4 = 1.0_c_double + 2.0_c_double**(-40)
  b4 = .true._c_bool
  call runtime%bind('rank', x4, y4, status)
  call require_ok()
  call runtime%run('rank', status)
  call require_ok()
  if (any(y4 /= x4)) error stop 'rank-4 float64 direct bind failed'
  call runtime%bind('rank', fortonnx_host_tensor(x4), fortonnx_host_tensor(y4), status)
  call require_ok()
  call runtime%run('rank', status)
  call require_ok()
  if (any(y4 /= x4)) error stop 'host tensor view failed'
  call runtime%close()
  call runtime%initialize(options, status)
  call require_ok()
  path = trim(model_path(:len_trim(model_path)-5)) // '_bool_4.onnx'
  call runtime%load('rank', trim(path), status)
  call require_ok()
  x4 = 1.0_c_double + 2.0_c_double**(-40)
  b4 = .true._c_bool
  call runtime%bind('rank', b4, c4, status)
  call require_ok()
  call runtime%run('rank', status)
  call require_ok()
  if (any(c4 .neqv. b4)) error stop 'rank-4 bool direct bind failed'
  call runtime%bind('rank', fortonnx_host_tensor(b4), fortonnx_host_tensor(c4), status)
  call require_ok()
  call runtime%run('rank', status)
  call require_ok()
  if (any(c4 .neqv. b4)) error stop 'host tensor view failed'
  call runtime%close()
  call runtime%initialize(options, status)
  call require_ok()
  path = trim(model_path(:len_trim(model_path)-5)) // '_compare_4.onnx'
  call runtime%load('rank', trim(path), status)
  call require_ok()
  x4 = 1.0_c_double + 2.0_c_double**(-40)
  b4 = .true._c_bool
  call runtime%bind('rank', x4, c4, status)
  call require_ok()
  call runtime%run('rank', status)
  call require_ok()
  if (any(.not. c4)) error stop 'rank-4 compare direct bind failed'
  call runtime%bind('rank', fortonnx_host_tensor(x4), fortonnx_host_tensor(c4), status)
  call require_ok()
  call runtime%run('rank', status)
  call require_ok()
  if (any(.not. c4)) error stop 'host tensor view failed'
  call runtime%close()
  call runtime%initialize(options, status)
  call require_ok()
  path = trim(model_path(:len_trim(model_path)-5)) // '_float64_5.onnx'
  call runtime%load('rank', trim(path), status)
  call require_ok()
  x5 = 1.0_c_double + 2.0_c_double**(-40)
  b5 = .true._c_bool
  call runtime%bind('rank', x5, y5, status)
  call require_ok()
  call runtime%run('rank', status)
  call require_ok()
  if (any(y5 /= x5)) error stop 'rank-5 float64 direct bind failed'
  call runtime%bind('rank', fortonnx_host_tensor(x5), fortonnx_host_tensor(y5), status)
  call require_ok()
  call runtime%run('rank', status)
  call require_ok()
  if (any(y5 /= x5)) error stop 'host tensor view failed'
  call runtime%close()
  call runtime%initialize(options, status)
  call require_ok()
  path = trim(model_path(:len_trim(model_path)-5)) // '_bool_5.onnx'
  call runtime%load('rank', trim(path), status)
  call require_ok()
  x5 = 1.0_c_double + 2.0_c_double**(-40)
  b5 = .true._c_bool
  call runtime%bind('rank', b5, c5, status)
  call require_ok()
  call runtime%run('rank', status)
  call require_ok()
  if (any(c5 .neqv. b5)) error stop 'rank-5 bool direct bind failed'
  call runtime%bind('rank', fortonnx_host_tensor(b5), fortonnx_host_tensor(c5), status)
  call require_ok()
  call runtime%run('rank', status)
  call require_ok()
  if (any(c5 .neqv. b5)) error stop 'host tensor view failed'
  call runtime%close()
  call runtime%initialize(options, status)
  call require_ok()
  path = trim(model_path(:len_trim(model_path)-5)) // '_compare_5.onnx'
  call runtime%load('rank', trim(path), status)
  call require_ok()
  x5 = 1.0_c_double + 2.0_c_double**(-40)
  b5 = .true._c_bool
  call runtime%bind('rank', x5, c5, status)
  call require_ok()
  call runtime%run('rank', status)
  call require_ok()
  if (any(.not. c5)) error stop 'rank-5 compare direct bind failed'
  call runtime%bind('rank', fortonnx_host_tensor(x5), fortonnx_host_tensor(c5), status)
  call require_ok()
  call runtime%run('rank', status)
  call require_ok()
  if (any(.not. c5)) error stop 'host tensor view failed'
  call runtime%close()
  call runtime%initialize(options, status)
  call require_ok()
  path = trim(model_path(:len_trim(model_path)-5)) // '_float64_6.onnx'
  call runtime%load('rank', trim(path), status)
  call require_ok()
  x6 = 1.0_c_double + 2.0_c_double**(-40)
  b6 = .true._c_bool
  call runtime%bind('rank', x6, y6, status)
  call require_ok()
  call runtime%run('rank', status)
  call require_ok()
  if (any(y6 /= x6)) error stop 'rank-6 float64 direct bind failed'
  call runtime%bind('rank', fortonnx_host_tensor(x6), fortonnx_host_tensor(y6), status)
  call require_ok()
  call runtime%run('rank', status)
  call require_ok()
  if (any(y6 /= x6)) error stop 'host tensor view failed'
  call runtime%close()
  call runtime%initialize(options, status)
  call require_ok()
  path = trim(model_path(:len_trim(model_path)-5)) // '_bool_6.onnx'
  call runtime%load('rank', trim(path), status)
  call require_ok()
  x6 = 1.0_c_double + 2.0_c_double**(-40)
  b6 = .true._c_bool
  call runtime%bind('rank', b6, c6, status)
  call require_ok()
  call runtime%run('rank', status)
  call require_ok()
  if (any(c6 .neqv. b6)) error stop 'rank-6 bool direct bind failed'
  call runtime%bind('rank', fortonnx_host_tensor(b6), fortonnx_host_tensor(c6), status)
  call require_ok()
  call runtime%run('rank', status)
  call require_ok()
  if (any(c6 .neqv. b6)) error stop 'host tensor view failed'
  call runtime%close()
  call runtime%initialize(options, status)
  call require_ok()
  path = trim(model_path(:len_trim(model_path)-5)) // '_compare_6.onnx'
  call runtime%load('rank', trim(path), status)
  call require_ok()
  x6 = 1.0_c_double + 2.0_c_double**(-40)
  b6 = .true._c_bool
  call runtime%bind('rank', x6, c6, status)
  call require_ok()
  call runtime%run('rank', status)
  call require_ok()
  if (any(.not. c6)) error stop 'rank-6 compare direct bind failed'
  call runtime%bind('rank', fortonnx_host_tensor(x6), fortonnx_host_tensor(c6), status)
  call require_ok()
  call runtime%run('rank', status)
  call require_ok()
  if (any(.not. c6)) error stop 'host tensor view failed'
  call runtime%close()
  call runtime%initialize(options, status)
  call require_ok()
  path = trim(model_path(:len_trim(model_path)-5)) // '_float64_7.onnx'
  call runtime%load('rank', trim(path), status)
  call require_ok()
  x7 = 1.0_c_double + 2.0_c_double**(-40)
  b7 = .true._c_bool
  call runtime%bind('rank', x7, y7, status)
  call require_ok()
  call runtime%run('rank', status)
  call require_ok()
  if (any(y7 /= x7)) error stop 'rank-7 float64 direct bind failed'
  call runtime%bind('rank', fortonnx_host_tensor(x7), fortonnx_host_tensor(y7), status)
  call require_ok()
  call runtime%run('rank', status)
  call require_ok()
  if (any(y7 /= x7)) error stop 'host tensor view failed'
  call runtime%close()
  call runtime%initialize(options, status)
  call require_ok()
  path = trim(model_path(:len_trim(model_path)-5)) // '_bool_7.onnx'
  call runtime%load('rank', trim(path), status)
  call require_ok()
  x7 = 1.0_c_double + 2.0_c_double**(-40)
  b7 = .true._c_bool
  call runtime%bind('rank', b7, c7, status)
  call require_ok()
  call runtime%run('rank', status)
  call require_ok()
  if (any(c7 .neqv. b7)) error stop 'rank-7 bool direct bind failed'
  call runtime%bind('rank', fortonnx_host_tensor(b7), fortonnx_host_tensor(c7), status)
  call require_ok()
  call runtime%run('rank', status)
  call require_ok()
  if (any(c7 .neqv. b7)) error stop 'host tensor view failed'
  call runtime%close()
  call runtime%initialize(options, status)
  call require_ok()
  path = trim(model_path(:len_trim(model_path)-5)) // '_compare_7.onnx'
  call runtime%load('rank', trim(path), status)
  call require_ok()
  x7 = 1.0_c_double + 2.0_c_double**(-40)
  b7 = .true._c_bool
  call runtime%bind('rank', x7, c7, status)
  call require_ok()
  call runtime%run('rank', status)
  call require_ok()
  if (any(.not. c7)) error stop 'rank-7 compare direct bind failed'
  call runtime%bind('rank', fortonnx_host_tensor(x7), fortonnx_host_tensor(c7), status)
  call require_ok()
  call runtime%run('rank', status)
  call require_ok()
  if (any(.not. c7)) error stop 'host tensor view failed'
  call runtime%close()
  call runtime%initialize(options, status)
  call require_ok()
  call runtime%close()
  path = trim(model_path(:len_trim(model_path)-5)) // '_float64_8.onnx'
  call runtime%initialize(options, status)
  call require_ok()
  call runtime%load('rank8', trim(path), status)
  call require_ok()
  flat_x = 1.0_c_double + 2.0_c_double**(-40)
  flat_b = .true._c_bool
  call runtime%bind('rank8', c_loc(flat_x), c_loc(flat_y), rank8_shape, rank8_shape, status, &
      input_type=fortonnx_float64, output_type=fortonnx_float64)
  call require_ok()
  call runtime%run('rank8', status)
  call require_ok()
  if (any(flat_y /= flat_x)) error stop 'rank-8 typed pointer failed'
  call runtime%bind('rank8', fortonnx_pointer_tensor(c_loc(flat_x), rank8_shape, fortonnx_cpu, &
      element_type=fortonnx_float64), fortonnx_pointer_tensor(c_loc(flat_y), rank8_shape, fortonnx_cpu, &
      element_type=fortonnx_float64), status)
  call require_ok()
  call runtime%run('rank8', status)
  call require_ok()
  if (any(flat_y /= flat_x)) error stop 'rank-8 pointer view failed'
  call runtime%close()
  path = trim(model_path(:len_trim(model_path)-5)) // '_bool_8.onnx'
  call runtime%initialize(options, status)
  call require_ok()
  call runtime%load('rank8', trim(path), status)
  call require_ok()
  flat_x = 1.0_c_double + 2.0_c_double**(-40)
  flat_b = .true._c_bool
  call runtime%bind('rank8', c_loc(flat_b), c_loc(flat_c), rank8_shape, rank8_shape, status, &
      input_type=fortonnx_bool, output_type=fortonnx_bool)
  call require_ok()
  call runtime%run('rank8', status)
  call require_ok()
  if (any(.not. flat_c)) error stop 'rank-8 typed pointer failed'
  call runtime%bind('rank8', fortonnx_pointer_tensor(c_loc(flat_b), rank8_shape, fortonnx_cpu, &
      element_type=fortonnx_bool), fortonnx_pointer_tensor(c_loc(flat_c), rank8_shape, fortonnx_cpu, &
      element_type=fortonnx_bool), status)
  call require_ok()
  call runtime%run('rank8', status)
  call require_ok()
  if (any(.not. flat_c)) error stop 'rank-8 pointer view failed'
  call runtime%close()
  path = trim(model_path(:len_trim(model_path)-5)) // '_compare_8.onnx'
  call runtime%initialize(options, status)
  call require_ok()
  call runtime%load('rank8', trim(path), status)
  call require_ok()
  flat_x = 1.0_c_double + 2.0_c_double**(-40)
  flat_b = .true._c_bool
  call runtime%bind('rank8', c_loc(flat_x), c_loc(flat_c), rank8_shape, rank8_shape, status, &
      input_type=fortonnx_float64, output_type=fortonnx_bool)
  call require_ok()
  call runtime%run('rank8', status)
  call require_ok()
  if (any(.not. flat_c)) error stop 'rank-8 typed pointer failed'
  call runtime%bind('rank8', fortonnx_pointer_tensor(c_loc(flat_x), rank8_shape, fortonnx_cpu, &
      element_type=fortonnx_float64), fortonnx_pointer_tensor(c_loc(flat_c), rank8_shape, fortonnx_cpu, &
      element_type=fortonnx_bool), status)
  call require_ok()
  call runtime%run('rank8', status)
  call require_ok()
  if (any(.not. flat_c)) error stop 'rank-8 pointer view failed'
  call runtime%close()
  print *, 'FortONNX FP64/bool CPU test passed (host ranks 1-7, pointer rank 8)'
contains
  subroutine require_ok()
    if (.not. status%ok()) then
      print *, status%message
      error stop 1
    end if
  end subroutine require_ok
  subroutine require_mismatch(direction, model_type, caller_type)
    character(len=*), intent(in) :: direction, model_type, caller_type
    if (status%ok()) error stop 'type mismatch accepted'
    if (index(status%message, direction) == 0 .or. &
        index(status%message, 'model '//model_type) == 0 .or. &
        index(status%message, 'caller '//caller_type) == 0) error stop 'unclear mismatch error'
  end subroutine require_mismatch
end program test_fp64_cpu
