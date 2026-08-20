program test_cpu
  use, intrinsic :: iso_c_binding
  use fortonnx
  implicit none

  integer, parameter :: batch_size = 4
  type(fortonnx_runtime) :: runtime
  type(fortonnx_options) :: options
  type(fortonnx_status) :: status
  real(c_float), target :: input(2, batch_size)
  real(c_float), target :: output(1, batch_size)
  real(c_float) :: expected(batch_size)
  character(len=1024) :: model_path
  integer(c_int64_t) :: input_features, output_features

  if (command_argument_count() /= 1) error stop 'test_cpu requires MODEL.onnx'
  call get_command_argument(1, model_path)

  input(:, 1) = [0.0_c_float, 0.0_c_float]
  input(:, 2) = [1.0_c_float, 0.0_c_float]
  input(:, 3) = [0.0_c_float, 1.0_c_float]
  input(:, 4) = [2.0_c_float, -1.0_c_float]
  output = 0.0_c_float
  expected = [0.5_c_float, 2.5_c_float, -2.5_c_float, 7.5_c_float]

  options%backend = fortonnx_cpu
  options%intra_op_threads = 2
  call runtime%initialize(options, status)
  call require_ok(status)
  call runtime%load('linear', trim(model_path), status)
  call require_ok(status)
  call runtime%features('linear', input_features, output_features, status)
  call require_ok(status)
  if (input_features /= 2 .or. output_features /= 1) error stop 'wrong feature counts'
  call runtime%bind('linear', input, output, status)
  call require_ok(status)
  call runtime%run('linear', status)
  call require_ok(status)

  if (maxval(abs(output(1, :) - expected)) > 1.0e-6_c_float) then
    write(*, '(a,4f10.4)') 'actual:   ', output(1, :)
    write(*, '(a,4f10.4)') 'expected: ', expected
    error stop 'incorrect inference result'
  end if
  if (runtime%size() /= 1) error stop 'wrong model count'
  call runtime%close()
  write(*, '(a)') 'FortONNX CPU test passed'

contains

  subroutine require_ok(current)
    type(fortonnx_status), intent(in) :: current
    if (.not. current%ok()) then
      write(*, '(a,i0,2a)') 'FortONNX error ', current%code, ': ', current%message
      error stop 1
    end if
  end subroutine require_ok
end program test_cpu
