program test_cnn_cpu
  use, intrinsic :: iso_c_binding
  use fortonnx
  implicit none

  integer, parameter :: batch_size = 2
  type(fortonnx_runtime) :: runtime
  type(fortonnx_options) :: options
  type(fortonnx_status) :: status
  real(c_float), target :: input(4, 4, 1, batch_size)
  real(c_float), target :: output(2, 2, 1, batch_size)
  real(c_float) :: expected(2, 2, 1, batch_size)
  integer(c_int64_t), allocatable :: input_shape(:), output_shape(:)
  character(len=1024) :: model_path
  integer :: value

  if (command_argument_count() /= 1) error stop 'test_cnn_cpu requires MODEL.onnx'
  call get_command_argument(1, model_path)

  input(:, :, 1, 1) = reshape([(real(value, c_float), value = 1, 16)], [4, 4])
  input(:, :, 1, 2) = 1.0_c_float
  output = 0.0_c_float
  expected = 0.0_c_float
  expected(:, :, 1, 1) = reshape([0.0_c_float, 3.0_c_float, &
      30.0_c_float, 39.0_c_float], [2, 2])

  options%backend = fortonnx_cpu
  options%intra_op_threads = 2
  call runtime%initialize(options, status)
  call require_ok(status)
  call runtime%load('cnn', trim(model_path), status)
  call require_ok(status)
  call runtime%shapes('cnn', input_shape, output_shape, status)
  call require_ok(status)
  if (any(input_shape /= [-1_c_int64_t, 1_c_int64_t, 4_c_int64_t, 4_c_int64_t])) &
      error stop 'wrong CNN input shape'
  if (any(output_shape /= [-1_c_int64_t, 1_c_int64_t, 2_c_int64_t, 2_c_int64_t])) &
      error stop 'wrong CNN output shape'

  ! Fortran (W,H,C,N) is the same contiguous storage as ONNX [N,C,H,W].
  call runtime%bind('cnn', input, output, status)
  call require_ok(status)
  call runtime%run('cnn', status)
  call require_ok(status)
  call require_values(output, expected)

  ! The pointer-plus-shape overload supports any positive tensor rank.
  output = 0.0_c_float
  call runtime%bind('cnn', c_loc(input), c_loc(output), &
      [2_c_int64_t, 1_c_int64_t, 4_c_int64_t, 4_c_int64_t], &
      [2_c_int64_t, 1_c_int64_t, 2_c_int64_t, 2_c_int64_t], status)
  call require_ok(status)
  call runtime%run('cnn', status)
  call require_ok(status)
  call require_values(output, expected)

  call runtime%close()
  write(*, '(a)') 'FortONNX CPU CNN test passed'

contains

  subroutine require_ok(current)
    type(fortonnx_status), intent(in) :: current
    if (.not. current%ok()) then
      write(*, '(a,i0,2a)') 'FortONNX error ', current%code, ': ', current%message
      error stop 1
    end if
  end subroutine require_ok

  subroutine require_values(actual, reference)
    real(c_float), intent(in) :: actual(:, :, :, :), reference(:, :, :, :)
    if (maxval(abs(actual - reference)) > 1.0e-6_c_float) then
      write(*, '(a,8f10.4)') 'actual:   ', actual
      write(*, '(a,8f10.4)') 'expected: ', reference
      error stop 'incorrect CNN inference result'
    end if
  end subroutine require_values
end program test_cnn_cpu
