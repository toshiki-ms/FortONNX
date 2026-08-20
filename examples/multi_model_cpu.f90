program multi_model_cpu
  use, intrinsic :: iso_c_binding
  use fortonnx
  implicit none

  integer, parameter :: batch_size = 1024
  type(fortonnx_runtime) :: runtime
  type(fortonnx_options) :: options
  type(fortonnx_status) :: status
  real(c_float), target :: input_a(2, batch_size), input_b(2, batch_size)
  real(c_float), target :: output_a(1, batch_size), output_b(1, batch_size)
  character(len=1024) :: model_path
  integer :: sample

  if (command_argument_count() /= 1) then
    write(*, '(a)') 'usage: multi_model_cpu MODEL.onnx'
    stop 2
  end if
  call get_command_argument(1, model_path)

  options%backend = fortonnx_cpu
  options%intra_op_threads = 4
  call runtime%initialize(options, status)
  call check(status)
  call runtime%load('atmosphere_a', trim(model_path), status)
  call check(status)
  call runtime%load('atmosphere_b', trim(model_path), status)
  call check(status)

  do sample = 1, batch_size
    input_a(:, sample) = [real(sample, c_float), 1.0_c_float]
    input_b(:, sample) = [-real(sample, c_float), 0.5_c_float]
  end do
  call runtime%bind('atmosphere_a', input_a, output_a, status)
  call check(status)
  call runtime%bind('atmosphere_b', input_b, output_b, status)
  call check(status)
  call runtime%run_all(status)
  call check(status)
  write(*, '(a,i0)') 'models: ', runtime%size()
  write(*, '(a,2f12.4)') 'first outputs: ', output_a(1, 1), output_b(1, 1)
  call runtime%close()

contains

  subroutine check(current)
    type(fortonnx_status), intent(in) :: current
    if (.not. current%ok()) then
      write(*, '(a)') current%message
      error stop 1
    end if
  end subroutine check
end program multi_model_cpu
