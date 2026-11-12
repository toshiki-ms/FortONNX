program test_cuda_graph
  ! options%cuda_graph: replayed runs read the bound buffers again; a model with an operator that a replay does not
  ! repeat is refused. CUDA keeps one graph per binding, a TensorRT model has one binding; a binding whose first run
  ! fails keeps none.
  use, intrinsic :: iso_c_binding
  use cudafor
  use fortonnx
  implicit none

  integer, parameter :: batch_size = 4, passes = 4
  type(fortonnx_runtime) :: runtime
  type(fortonnx_options) :: options
  type(fortonnx_status) :: status
  type(fortonnx_cuda_stream) :: stream
  real(c_float), allocatable, target :: input_host(:, :), output_host(:, :)
  real(c_float), allocatable, device, target :: input_device(:, :), output_device(:, :)
  real(c_float), allocatable, device, target :: second_input(:, :), second_output(:, :)
  real(c_float), allocatable, device, target :: long_output(:, :)
  real(c_float) :: expected(batch_size)
  character(len=1024) :: model_path, gather_path, random_path, dropout_path
  character(len=32) :: backend
  logical :: tensorrt
  integer :: pass

  if (command_argument_count() /= 5) then
    error stop 'test_cuda_graph requires LINEAR.onnx GATHER_ND.onnx RANDOM.onnx DROPOUT_TRAINING.onnx BACKEND'
  end if
  call get_command_argument(1, model_path)
  call get_command_argument(2, gather_path)
  call get_command_argument(3, random_path)
  call get_command_argument(4, dropout_path)
  call get_command_argument(5, backend)
  select case (trim(backend))
  case ('cuda')
    options%backend = fortonnx_cuda
  case ('tensorrt')
    options%backend = fortonnx_tensorrt
  case default
    error stop 'BACKEND must be cuda or tensorrt'
  end select
  tensorrt = options%backend == fortonnx_tensorrt

  allocate(input_host(2, batch_size), output_host(1, batch_size))
  allocate(input_device(2, batch_size), output_device(1, batch_size))
  allocate(second_input(2, batch_size), second_output(1, batch_size), long_output(1, batch_size + 1))

  call stream%create(status)
  call require_ok(status)
  options%user_compute_stream = stream%c_pointer()
  options%use_tf32 = .false.
  options%cuda_graph = .true.
  call runtime%initialize(options, status)
  call require_ok(status)
  call runtime%load('linear', trim(model_path), status)
  call require_ok(status)

  ! An output buffer of another batch size: the run fails, and the binding does not take the place of a graph
  ! (the one place of a TensorRT model stays free for the buffers below).
  input_device = 0.0_c_float
  call runtime%bind('linear', input_device, long_output, status)
  call require_ok(status)
  call runtime%run('linear', status)
  if (status%ok()) error stop 'a run with an output buffer of another batch size succeeded'

  ! The first run of a binding captures its graph; the later ones replay it with new buffer contents.
  ! With CUDA a second set of buffers takes a graph of its own.
  do pass = 1, passes
    input_host(1, :) = [0.0_c_float, 1.0_c_float, 0.0_c_float, 2.0_c_float] * real(pass, c_float)
    input_host(2, :) = [0.0_c_float, 0.0_c_float, 1.0_c_float, -1.0_c_float] + real(pass, c_float)
    expected = 2.0_c_float * input_host(1, :) - 3.0_c_float * input_host(2, :) + 0.5_c_float
    input_device = input_host
    output_device = 0.0_c_float
    call runtime%bind('linear', input_device, output_device, status)
    call require_ok(status)
    call run_and_compare(output_device, 'first buffers')
    if (tensorrt) cycle
    second_input = input_host
    second_output = 0.0_c_float
    call runtime%bind('linear', second_input, second_output, status)
    call require_ok(status)
    call run_and_compare(second_output, 'second buffers')
  end do

  if (tensorrt) then
    ! The one graph of a TensorRT model belongs to the first buffers: other buffers are an error, not a replay.
    call runtime%bind('linear', second_input, second_output, status)
    call require_ok(status)
    call runtime%run('linear', status)
    if (status%ok()) error stop 'cuda_graph ran a second binding of a TensorRT model'
    call require_message('one captured graph')
  else
    call runtime%load('gather_nd', trim(gather_path), status)
    if (status%ok()) error stop 'cuda_graph accepted a model with GatherND'
    call require_message('GatherND')
  end if
  call runtime%load('random', trim(random_path), status)
  if (status%ok()) error stop 'cuda_graph accepted a model with a random-number operator'
  call require_message('RandomNormalLike')
  call runtime%load('dropout', trim(dropout_path), status)
  if (status%ok()) error stop 'cuda_graph accepted a model with a dropout operator in training mode'
  call require_message('BitmaskDropout node with a training_mode input')

  call runtime%close()
  call stream%close(status)
  call require_ok(status)
  deallocate(input_device, output_device, second_input, second_output, long_output, input_host, output_host)
  write(*, '(a)') 'FortONNX cuda_graph test passed'

contains

  subroutine run_and_compare(result_device, label)
    real(c_float), device, intent(in) :: result_device(:, :)
    character(len=*), intent(in) :: label
    call runtime%run('linear', status)
    call require_ok(status)
    call stream%synchronize(status)
    call require_ok(status)
    output_host = result_device
    if (maxval(abs(output_host(1, :) - expected)) > 1.0e-5_c_float) then
      write(*, '(2a,i0)') label, ', pass ', pass
      write(*, '(a,4f10.4)') 'actual:   ', output_host(1, :)
      write(*, '(a,4f10.4)') 'expected: ', expected
      error stop 'incorrect result with cuda_graph'
    end if
  end subroutine run_and_compare

  subroutine require_message(text)
    character(len=*), intent(in) :: text
    if (index(status%message, text) == 0) then
      write(*, '(2a)') 'unexpected message: ', trim(status%message)
      error stop 1
    end if
  end subroutine require_message

  subroutine require_ok(current)
    type(fortonnx_status), intent(in) :: current
    if (.not. current%ok()) then
      write(*, '(a,i0,2a)') 'FortONNX error ', current%code, ': ', current%message
      error stop 1
    end if
  end subroutine require_ok
end program test_cuda_graph
