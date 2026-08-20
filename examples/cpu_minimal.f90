program cpu_minimal
  use, intrinsic :: iso_c_binding
  use fortonnx
  implicit none

  type(fortonnx_runtime) :: runtime
  type(fortonnx_options) :: options
  type(fortonnx_status) :: status
  real(c_float), target :: input(2, 3), output(1, 3)
  character(len=1024) :: model_path

  if (command_argument_count() /= 1) then
    write(*, '(a)') 'usage: cpu_minimal MODEL.onnx'
    stop 2
  end if
  call get_command_argument(1, model_path)

  options%backend = fortonnx_cpu
  options%intra_op_threads = 4
  call runtime%initialize(options, status)
  call check(status)
  call runtime%load('surrogate', trim(model_path), status)
  call check(status)

  ! ONNX sees [batch, features]; the Fortran layout is (features, batch).
  input(:, 1) = [0.0_c_float, 0.0_c_float]
  input(:, 2) = [1.0_c_float, 2.0_c_float]
  input(:, 3) = [-1.0_c_float, 0.5_c_float]
  output = 0.0_c_float
  call runtime%bind('surrogate', input, output, status)
  call check(status)
  call runtime%run('surrogate', status)
  call check(status)
  write(*, '(3f12.6)') output
  call runtime%close()

contains

  subroutine check(current)
    type(fortonnx_status), intent(in) :: current
    if (.not. current%ok()) then
      write(*, '(a)') current%message
      error stop 1
    end if
  end subroutine check
end program cpu_minimal
