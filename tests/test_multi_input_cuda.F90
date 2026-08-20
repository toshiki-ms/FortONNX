program test_multi_input_cuda
  use, intrinsic :: iso_c_binding
  use cudafor
  use fortonnx
  implicit none

  integer, parameter :: batch_size = 2
  type(fortonnx_runtime) :: runtime
  type(fortonnx_options) :: options
  type(fortonnx_status) :: status
  type(fortonnx_cuda_stream) :: stream
  type(fortonnx_tensor) :: inputs(2), outputs(2)
  real(c_float), target :: image_host(2, 2, 1, batch_size)
  real(c_float), target :: enhanced_host(2, 2, 1, batch_size)
  real(c_float), target :: scaled_host(2, 2, 1, batch_size)
  real(c_float), device, target :: image_device(2, 2, 1, batch_size)
  real(c_float), device, target :: gain_device(1)
  real(c_float), device, target :: enhanced_device(2, 2, 1, batch_size)
  real(c_float), device, target :: scaled_device(2, 2, 1, batch_size)
  real(c_float) :: expected(2, 2, 1, batch_size)
  character(len=1024) :: model_path, backend

  if (command_argument_count() /= 2) &
      error stop 'test_multi_input_cuda requires MODEL.onnx BACKEND'
  call get_command_argument(1, model_path)
  call get_command_argument(2, backend)

  image_host(:, :, 1, 1) = reshape([-1.0_c_float, 0.0_c_float, &
      1.0_c_float, 2.0_c_float], [2, 2])
  image_host(:, :, 1, 2) = reshape([3.0_c_float, 4.0_c_float, &
      5.0_c_float, 6.0_c_float], [2, 2])
  expected = max(2.0_c_float * image_host - 1.0_c_float, 0.0_c_float)
  image_device = image_host
  gain_device = 2.0_c_float
  enhanced_device = 0.0_c_float
  scaled_device = 0.0_c_float

  call stream%create(status)
  call require_ok(status)
  select case (trim(backend))
  case ('cuda')
    options%backend = fortonnx_cuda
  case ('tensorrt')
    options%backend = fortonnx_tensorrt
  case default
    error stop 'backend must be cuda or tensorrt'
  end select
  options%user_compute_stream = stream%c_pointer()
  options%use_tf32 = .false.
  call runtime%initialize(options, status)
  call require_ok(status)
  call runtime%load('controlled_image', trim(model_path), status)
  call require_ok(status)

  inputs(1) = fortonnx_device_tensor(gain_device, 'gain')
  inputs(2) = fortonnx_device_tensor(image_device, 'image')
  outputs(1) = fortonnx_device_tensor(scaled_device, 'scaled')
  outputs(2) = fortonnx_device_tensor(enhanced_device, 'enhanced')
  call runtime%bind('controlled_image', inputs, outputs, status)
  call require_ok(status)
  call runtime%run('controlled_image', status)
  call require_ok(status)
  call stream%synchronize(status)
  call require_ok(status)
  enhanced_host = enhanced_device
  scaled_host = scaled_device
  if (maxval(abs(enhanced_host - expected)) > 1.0e-5_c_float) then
    error stop 'incorrect multi-input GPU inference result'
  end if
  if (maxval(abs(scaled_host - 2.0_c_float * image_host)) > 1.0e-5_c_float) then
    error stop 'incorrect multi-output GPU inference result'
  end if

  call runtime%close()
  call stream%close(status)
  call require_ok(status)
  write(*, '(3a)') 'FortONNX ', trim(backend), ' multi-input test passed'

contains

  subroutine require_ok(current)
    type(fortonnx_status), intent(in) :: current
    if (.not. current%ok()) then
      write(*, '(a,i0,2a)') 'FortONNX error ', current%code, ': ', current%message
      error stop 1
    end if
  end subroutine require_ok
end program test_multi_input_cuda
