program test_cuda
  use, intrinsic :: iso_c_binding
  use cudafor
  use fortonnx
  implicit none

  integer, parameter :: batch_size = 4
  type(fortonnx_runtime) :: runtime
  type(fortonnx_options) :: options
  type(fortonnx_status) :: status
  type(fortonnx_cuda_stream) :: stream
  real(c_float), allocatable, target :: input_host(:, :), output_host(:, :)
  real(c_float), allocatable, device, target :: input_device(:, :), output_device(:, :)
  real(c_float) :: expected(batch_size)
  character(len=1024) :: model_path, backend

  if (command_argument_count() /= 2) error stop 'test_cuda requires MODEL.onnx BACKEND'
  call get_command_argument(1, model_path)
  call get_command_argument(2, backend)

  allocate(input_host(2, batch_size), output_host(1, batch_size))
  allocate(input_device(2, batch_size), output_device(1, batch_size))
  input_host(:, 1) = [0.0_c_float, 0.0_c_float]
  input_host(:, 2) = [1.0_c_float, 0.0_c_float]
  input_host(:, 3) = [0.0_c_float, 1.0_c_float]
  input_host(:, 4) = [2.0_c_float, -1.0_c_float]
  expected = [0.5_c_float, 2.5_c_float, -2.5_c_float, 7.5_c_float]
  input_device = input_host
  output_device = 0.0_c_float

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
  call runtime%load('linear', trim(model_path), status)
  call require_ok(status)
  call runtime%bind('linear', input_device, output_device, status)
  call require_ok(status)
  call runtime%run('linear', status)
  call require_ok(status)
  call stream%synchronize(status)
  call require_ok(status)
  output_host = output_device
  if (maxval(abs(output_host(1, :) - expected)) > 1.0e-5_c_float) then
    write(*, '(a,4f10.4)') 'actual:   ', output_host(1, :)
    write(*, '(a,4f10.4)') 'expected: ', expected
    error stop 'incorrect GPU inference result'
  end if

  call runtime%close()
  call stream%close(status)
  call require_ok(status)
  deallocate(input_device, output_device, input_host, output_host)
  write(*, '(3a)') 'FortONNX ', trim(backend), ' test passed'

contains

  subroutine require_ok(current)
    type(fortonnx_status), intent(in) :: current
    if (.not. current%ok()) then
      write(*, '(a,i0,2a)') 'FortONNX error ', current%code, ': ', current%message
      error stop 1
    end if
  end subroutine require_ok
end program test_cuda
