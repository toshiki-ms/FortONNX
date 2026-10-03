program test_fp64_cuda
  use, intrinsic :: iso_c_binding
  use cudafor
  use fortonnx
  implicit none
  type(fortonnx_runtime) :: runtime
  type(fortonnx_options) :: options
  type(fortonnx_status) :: status
  type(fortonnx_tensor) :: inputs(2), outputs(2)
  real(c_double), target :: state(2,3), values(2,3), expected(2,3)
  logical(c_bool), target :: enabled(2,3), valid(2,3)
  real(c_float), target :: wrong(2,3)
  integer(c_int) :: element_type
  character(len=1024) :: model_path, path
  type(fortonnx_cuda_stream) :: stream
  real(c_double), device, target :: state_device(2,3), values_device(2,3)
  logical(c_bool), device, target :: enabled_device(2,3), valid_device(2,3)
  real(c_float), device, target :: wrong_device(2,3)
  character(len=32) :: backend
  real(c_double) :: host1(2)
  logical(c_bool) :: flag1(2)
  real(c_double) :: host2(2,2)
  logical(c_bool) :: flag2(2,2)
  real(c_double) :: host3(2,2,2)
  logical(c_bool) :: flag3(2,2,2)
  real(c_double) :: host4(2,2,2,2)
  logical(c_bool) :: flag4(2,2,2,2)
  real(c_double) :: host5(2,2,2,2,2)
  logical(c_bool) :: flag5(2,2,2,2,2)
  real(c_double) :: host6(2,2,2,2,2,2)
  logical(c_bool) :: flag6(2,2,2,2,2,2)
  real(c_double) :: host7(2,2,2,2,2,2,2)
  logical(c_bool) :: flag7(2,2,2,2,2,2,2)
  real(c_double), device, target :: x1(2), y1(2)
  logical(c_bool), device, target :: b1(2), c1(2)
  real(c_double), device, target :: x2(2,2), y2(2,2)
  logical(c_bool), device, target :: b2(2,2), c2(2,2)
  real(c_double), device, target :: x3(2,2,2), y3(2,2,2)
  logical(c_bool), device, target :: b3(2,2,2), c3(2,2,2)
  real(c_double), device, target :: x4(2,2,2,2), y4(2,2,2,2)
  logical(c_bool), device, target :: b4(2,2,2,2), c4(2,2,2,2)
  real(c_double), device, target :: x5(2,2,2,2,2), y5(2,2,2,2,2)
  logical(c_bool), device, target :: b5(2,2,2,2,2), c5(2,2,2,2,2)
  real(c_double), device, target :: x6(2,2,2,2,2,2), y6(2,2,2,2,2,2)
  logical(c_bool), device, target :: b6(2,2,2,2,2,2), c6(2,2,2,2,2,2)
  real(c_double), device, target :: x7(2,2,2,2,2,2,2), y7(2,2,2,2,2,2,2)
  logical(c_bool), device, target :: b7(2,2,2,2,2,2,2), c7(2,2,2,2,2,2,2)

  if (command_argument_count() /= 2) error stop 'requires MODEL.onnx BACKEND'
  call get_command_argument(1, model_path)
  call get_command_argument(2, backend)
  select case (trim(backend))
  case ('cuda')
    options%backend = fortonnx_cuda
  case ('tensorrt')
    options%backend = fortonnx_tensorrt
  case default
    error stop 'backend must be cuda or tensorrt'
  end select
  call stream%create(status)
  call require_ok()
  options%user_compute_stream = stream%c_pointer()
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
  state = reshape([1.0_c_double + 2.0_c_double**(-40), -1.0_c_double, &
      2.0_c_double**(-44), 2.0_c_double, -2.0_c_double, 0.0_c_double], [2,3])
  enabled = reshape([.true._c_bool, .false._c_bool, .true._c_bool, &
      .false._c_bool, .true._c_bool, .false._c_bool], [2,3])
  expected = state
  where (enabled) expected = state + 2.0_c_double**(-42)
  state_device = state
  enabled_device = enabled
  inputs(1) = fortonnx_device_tensor(enabled_device, 'enabled')
  inputs(2) = fortonnx_device_tensor(state_device, 'state')
  outputs(1) = fortonnx_device_tensor(valid_device, 'valid')
  outputs(2) = fortonnx_device_tensor(values_device, 'values')
  call runtime%bind('mixed', inputs, outputs, status)
  call require_ok()
  call runtime%run('mixed', status)
  call require_ok()
  call stream%synchronize(status)
  call require_ok()
  values = values_device
  valid = valid_device
  if (any(values /= expected)) error stop 'FP64 precision lost'
  if (any(valid .neqv. (enabled .and. state > 0.0_c_double))) error stop 'wrong bool output'
  inputs(2) = fortonnx_device_tensor(wrong_device, 'state')
  call runtime%bind('mixed', inputs, outputs, status)
  call require_mismatch('input', 'float64', 'float32')
  inputs(2) = fortonnx_device_tensor(state_device, 'state')
  outputs(1) = fortonnx_device_tensor(wrong_device, 'valid')
  call runtime%bind('mixed', inputs, outputs, status)
  call require_mismatch('output', 'bool', 'float32')
  outputs(1) = fortonnx_pointer_tensor(transfer(c_devloc(valid_device), c_null_ptr), [3_c_int64_t,2_c_int64_t], &
      fortonnx_cuda, 'valid', element_type=fortonnx_bool)
  inputs(2) = fortonnx_pointer_tensor(transfer(c_devloc(state_device), c_null_ptr), [3_c_int64_t,2_c_int64_t], &
      fortonnx_cuda, 'state', element_type=fortonnx_float64)
  call runtime%bind('mixed', inputs, outputs, status)
  call require_ok()
  call runtime%run('mixed', status)
  call require_ok()
  call stream%synchronize(status)
  call require_ok()
  values = values_device
  valid = valid_device
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
  call stream%synchronize(status)
  call require_ok()
  host1 = y1
  if (any(host1 /= 1.0_c_double + 2.0_c_double**(-40))) error stop "FP64 device rank failure"
  call runtime%bind('rank', fortonnx_device_tensor(x1), fortonnx_device_tensor(y1), status)
  call require_ok()
  call runtime%run('rank', status)
  call require_ok()
  call stream%synchronize(status)
  call require_ok()
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
  call stream%synchronize(status)
  call require_ok()
  flag1 = c1
  if (any(.not. flag1)) error stop "bool device rank failure"
  call runtime%bind('rank', fortonnx_device_tensor(b1), fortonnx_device_tensor(c1), status)
  call require_ok()
  call runtime%run('rank', status)
  call require_ok()
  call stream%synchronize(status)
  call require_ok()
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
  call stream%synchronize(status)
  call require_ok()
  flag1 = c1
  if (any(.not. flag1)) error stop "bool device rank failure"
  call runtime%bind('rank', fortonnx_device_tensor(x1), fortonnx_device_tensor(c1), status)
  call require_ok()
  call runtime%run('rank', status)
  call require_ok()
  call stream%synchronize(status)
  call require_ok()
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
  call stream%synchronize(status)
  call require_ok()
  host2 = y2
  if (any(host2 /= 1.0_c_double + 2.0_c_double**(-40))) error stop "FP64 device rank failure"
  call runtime%bind('rank', fortonnx_device_tensor(x2), fortonnx_device_tensor(y2), status)
  call require_ok()
  call runtime%run('rank', status)
  call require_ok()
  call stream%synchronize(status)
  call require_ok()
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
  call stream%synchronize(status)
  call require_ok()
  flag2 = c2
  if (any(.not. flag2)) error stop "bool device rank failure"
  call runtime%bind('rank', fortonnx_device_tensor(b2), fortonnx_device_tensor(c2), status)
  call require_ok()
  call runtime%run('rank', status)
  call require_ok()
  call stream%synchronize(status)
  call require_ok()
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
  call stream%synchronize(status)
  call require_ok()
  flag2 = c2
  if (any(.not. flag2)) error stop "bool device rank failure"
  call runtime%bind('rank', fortonnx_device_tensor(x2), fortonnx_device_tensor(c2), status)
  call require_ok()
  call runtime%run('rank', status)
  call require_ok()
  call stream%synchronize(status)
  call require_ok()
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
  call stream%synchronize(status)
  call require_ok()
  host3 = y3
  if (any(host3 /= 1.0_c_double + 2.0_c_double**(-40))) error stop "FP64 device rank failure"
  call runtime%bind('rank', fortonnx_device_tensor(x3), fortonnx_device_tensor(y3), status)
  call require_ok()
  call runtime%run('rank', status)
  call require_ok()
  call stream%synchronize(status)
  call require_ok()
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
  call stream%synchronize(status)
  call require_ok()
  flag3 = c3
  if (any(.not. flag3)) error stop "bool device rank failure"
  call runtime%bind('rank', fortonnx_device_tensor(b3), fortonnx_device_tensor(c3), status)
  call require_ok()
  call runtime%run('rank', status)
  call require_ok()
  call stream%synchronize(status)
  call require_ok()
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
  call stream%synchronize(status)
  call require_ok()
  flag3 = c3
  if (any(.not. flag3)) error stop "bool device rank failure"
  call runtime%bind('rank', fortonnx_device_tensor(x3), fortonnx_device_tensor(c3), status)
  call require_ok()
  call runtime%run('rank', status)
  call require_ok()
  call stream%synchronize(status)
  call require_ok()
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
  call stream%synchronize(status)
  call require_ok()
  host4 = y4
  if (any(host4 /= 1.0_c_double + 2.0_c_double**(-40))) error stop "FP64 device rank failure"
  call runtime%bind('rank', fortonnx_device_tensor(x4), fortonnx_device_tensor(y4), status)
  call require_ok()
  call runtime%run('rank', status)
  call require_ok()
  call stream%synchronize(status)
  call require_ok()
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
  call stream%synchronize(status)
  call require_ok()
  flag4 = c4
  if (any(.not. flag4)) error stop "bool device rank failure"
  call runtime%bind('rank', fortonnx_device_tensor(b4), fortonnx_device_tensor(c4), status)
  call require_ok()
  call runtime%run('rank', status)
  call require_ok()
  call stream%synchronize(status)
  call require_ok()
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
  call stream%synchronize(status)
  call require_ok()
  flag4 = c4
  if (any(.not. flag4)) error stop "bool device rank failure"
  call runtime%bind('rank', fortonnx_device_tensor(x4), fortonnx_device_tensor(c4), status)
  call require_ok()
  call runtime%run('rank', status)
  call require_ok()
  call stream%synchronize(status)
  call require_ok()
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
  call stream%synchronize(status)
  call require_ok()
  host5 = y5
  if (any(host5 /= 1.0_c_double + 2.0_c_double**(-40))) error stop "FP64 device rank failure"
  call runtime%bind('rank', fortonnx_device_tensor(x5), fortonnx_device_tensor(y5), status)
  call require_ok()
  call runtime%run('rank', status)
  call require_ok()
  call stream%synchronize(status)
  call require_ok()
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
  call stream%synchronize(status)
  call require_ok()
  flag5 = c5
  if (any(.not. flag5)) error stop "bool device rank failure"
  call runtime%bind('rank', fortonnx_device_tensor(b5), fortonnx_device_tensor(c5), status)
  call require_ok()
  call runtime%run('rank', status)
  call require_ok()
  call stream%synchronize(status)
  call require_ok()
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
  call stream%synchronize(status)
  call require_ok()
  flag5 = c5
  if (any(.not. flag5)) error stop "bool device rank failure"
  call runtime%bind('rank', fortonnx_device_tensor(x5), fortonnx_device_tensor(c5), status)
  call require_ok()
  call runtime%run('rank', status)
  call require_ok()
  call stream%synchronize(status)
  call require_ok()
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
  call stream%synchronize(status)
  call require_ok()
  host6 = y6
  if (any(host6 /= 1.0_c_double + 2.0_c_double**(-40))) error stop "FP64 device rank failure"
  call runtime%bind('rank', fortonnx_device_tensor(x6), fortonnx_device_tensor(y6), status)
  call require_ok()
  call runtime%run('rank', status)
  call require_ok()
  call stream%synchronize(status)
  call require_ok()
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
  call stream%synchronize(status)
  call require_ok()
  flag6 = c6
  if (any(.not. flag6)) error stop "bool device rank failure"
  call runtime%bind('rank', fortonnx_device_tensor(b6), fortonnx_device_tensor(c6), status)
  call require_ok()
  call runtime%run('rank', status)
  call require_ok()
  call stream%synchronize(status)
  call require_ok()
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
  call stream%synchronize(status)
  call require_ok()
  flag6 = c6
  if (any(.not. flag6)) error stop "bool device rank failure"
  call runtime%bind('rank', fortonnx_device_tensor(x6), fortonnx_device_tensor(c6), status)
  call require_ok()
  call runtime%run('rank', status)
  call require_ok()
  call stream%synchronize(status)
  call require_ok()
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
  call stream%synchronize(status)
  call require_ok()
  host7 = y7
  if (any(host7 /= 1.0_c_double + 2.0_c_double**(-40))) error stop "FP64 device rank failure"
  call runtime%bind('rank', fortonnx_device_tensor(x7), fortonnx_device_tensor(y7), status)
  call require_ok()
  call runtime%run('rank', status)
  call require_ok()
  call stream%synchronize(status)
  call require_ok()
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
  call stream%synchronize(status)
  call require_ok()
  flag7 = c7
  if (any(.not. flag7)) error stop "bool device rank failure"
  call runtime%bind('rank', fortonnx_device_tensor(b7), fortonnx_device_tensor(c7), status)
  call require_ok()
  call runtime%run('rank', status)
  call require_ok()
  call stream%synchronize(status)
  call require_ok()
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
  call stream%synchronize(status)
  call require_ok()
  flag7 = c7
  if (any(.not. flag7)) error stop "bool device rank failure"
  call runtime%bind('rank', fortonnx_device_tensor(x7), fortonnx_device_tensor(c7), status)
  call require_ok()
  call runtime%run('rank', status)
  call require_ok()
  call stream%synchronize(status)
  call require_ok()
  call runtime%close()
  call runtime%initialize(options, status)
  call require_ok()
  call runtime%close()
  call stream%close(status)
  call require_ok()
  print *, 'FortONNX FP64/bool GPU test passed (device ranks 1-7): ', trim(backend)
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
end program test_fp64_cuda
