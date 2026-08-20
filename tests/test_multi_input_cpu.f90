program test_multi_input_cpu
  use, intrinsic :: iso_c_binding
  use fortonnx
  implicit none

  integer, parameter :: batch_size = 2
  type(fortonnx_runtime) :: runtime
  type(fortonnx_options) :: options
  type(fortonnx_status) :: status
  type(fortonnx_tensor) :: inputs(2), outputs(2)
  real(c_float), target :: image(2, 2, 1, batch_size)
  real(c_float), target :: gain(1)
  real(c_float), target :: enhanced(2, 2, 1, batch_size)
  real(c_float), target :: scaled(2, 2, 1, batch_size)
  real(c_float) :: expected(2, 2, 1, batch_size)
  integer(c_int64_t), allocatable :: tensor_shape(:)
  character(len=1024) :: model_path
  integer :: input_count, output_count

  if (command_argument_count() /= 1) error stop 'test_multi_input_cpu requires MODEL.onnx'
  call get_command_argument(1, model_path)

  image(:, :, 1, 1) = reshape([-1.0_c_float, 0.0_c_float, &
      1.0_c_float, 2.0_c_float], [2, 2])
  image(:, :, 1, 2) = reshape([3.0_c_float, 4.0_c_float, &
      5.0_c_float, 6.0_c_float], [2, 2])
  gain = 2.0_c_float
  enhanced = 0.0_c_float
  scaled = 0.0_c_float
  expected = max(2.0_c_float * image - 1.0_c_float, 0.0_c_float)

  options%backend = fortonnx_cpu
  call runtime%initialize(options, status)
  call require_ok(status)
  call runtime%load('controlled_image', trim(model_path), status)
  call require_ok(status)
  call runtime%io_counts('controlled_image', input_count, output_count, status)
  call require_ok(status)
  if (input_count /= 2 .or. output_count /= 2) error stop 'wrong multi-I/O counts'
  call runtime%tensor_shape('controlled_image', fortonnx_input, 1, tensor_shape, status)
  call require_ok(status)
  if (any(tensor_shape /= [-1_c_int64_t, 1_c_int64_t, 2_c_int64_t, 2_c_int64_t])) &
      error stop 'wrong image input shape'
  call runtime%tensor_shape('controlled_image', fortonnx_input, 2, tensor_shape, status)
  call require_ok(status)
  if (any(tensor_shape /= [1_c_int64_t])) error stop 'wrong gain input shape'

  ! Deliberately reverse model input order; names provide the mapping.
  inputs(1) = fortonnx_host_tensor(gain, 'gain')
  inputs(2) = fortonnx_host_tensor(image, 'image')
  outputs(1) = fortonnx_host_tensor(scaled, 'scaled')
  outputs(2) = fortonnx_host_tensor(enhanced, 'enhanced')
  call runtime%bind('controlled_image', inputs, outputs, status)
  call require_ok(status)
  call runtime%run('controlled_image', status)
  call require_ok(status)
  if (maxval(abs(enhanced - expected)) > 1.0e-6_c_float) then
    error stop 'incorrect multi-input CPU inference result'
  end if
  if (maxval(abs(scaled - 2.0_c_float * image)) > 1.0e-6_c_float) then
    error stop 'incorrect multi-output CPU inference result'
  end if

  ! Unnamed views fall back to the ONNX input/output order.
  enhanced = 0.0_c_float
  scaled = 0.0_c_float
  inputs(1) = fortonnx_host_tensor(image)
  inputs(2) = fortonnx_host_tensor(gain)
  outputs(1) = fortonnx_host_tensor(enhanced)
  outputs(2) = fortonnx_host_tensor(scaled)
  call runtime%bind('controlled_image', inputs, outputs, status)
  call require_ok(status)
  call runtime%run('controlled_image', status)
  call require_ok(status)
  if (maxval(abs(enhanced - expected)) > 1.0e-6_c_float .or. &
      maxval(abs(scaled - 2.0_c_float * image)) > 1.0e-6_c_float) then
    error stop 'incorrect positional multi-I/O inference result'
  end if

  call runtime%close()
  write(*, '(a)') 'FortONNX CPU multi-input test passed'

contains

  subroutine require_ok(current)
    type(fortonnx_status), intent(in) :: current
    if (.not. current%ok()) then
      write(*, '(a,i0,2a)') 'FortONNX error ', current%code, ': ', current%message
      error stop 1
    end if
  end subroutine require_ok
end program test_multi_input_cpu
