module fortonnx_c_api
  use, intrinsic :: iso_c_binding
  implicit none
  private

  public :: c_fortonnx_cpu_context_create
  public :: c_fortonnx_cpu_context_destroy
  public :: c_fortonnx_session_create_cpu
  public :: c_fortonnx_session_create_cpu_with_custom_ops
  public :: c_fortonnx_session_create_gpu
  public :: c_fortonnx_session_create_gpu_with_custom_ops
  public :: c_fortonnx_session_get_features
  public :: c_fortonnx_session_get_tensor_count
  public :: c_fortonnx_session_get_tensor_rank
  public :: c_fortonnx_session_get_tensor_shape
  public :: c_fortonnx_session_get_tensor_rank_at
  public :: c_fortonnx_session_get_tensor_shape_at
  public :: c_fortonnx_session_find_tensor
  public :: c_fortonnx_session_begin_bind
  public :: c_fortonnx_session_bind_input
  public :: c_fortonnx_session_bind_input_typed
  public :: c_fortonnx_session_bind_output
  public :: c_fortonnx_session_bind_output_typed
  public :: c_fortonnx_session_bind
  public :: c_fortonnx_session_bind_tensor
  public :: c_fortonnx_session_bind_tensor_typed
  public :: c_fortonnx_session_run
  public :: c_fortonnx_session_destroy
  public :: c_fortonnx_last_error

  public :: c_fortonnx_session_get_tensor_type_at

  interface
    function c_fortonnx_session_get_tensor_type_at(session, is_input, index, element_type) &
        bind(C, name='fortonnx_session_get_tensor_type_at') result(code)
      import :: c_int, c_int64_t, c_ptr
      type(c_ptr), value :: session
      integer(c_int), value :: is_input
      integer(c_int64_t), value :: index
      integer(c_int), intent(out) :: element_type
      integer(c_int) :: code
    end function c_fortonnx_session_get_tensor_type_at

    function c_fortonnx_cpu_context_create(intra_threads, inter_threads) &
        bind(C, name='fortonnx_cpu_context_create') result(context)
      import :: c_int, c_ptr
      integer(c_int), value :: intra_threads, inter_threads
      type(c_ptr) :: context
    end function c_fortonnx_cpu_context_create

    subroutine c_fortonnx_cpu_context_destroy(context) &
        bind(C, name='fortonnx_cpu_context_destroy')
      import :: c_ptr
      type(c_ptr), value :: context
    end subroutine c_fortonnx_cpu_context_destroy

    function c_fortonnx_session_create_cpu(model_path, context) &
        bind(C, name='fortonnx_session_create_cpu') result(session)
      import :: c_char, c_ptr
      character(kind=c_char), intent(in) :: model_path(*)
      type(c_ptr), value :: context
      type(c_ptr) :: session
    end function c_fortonnx_session_create_cpu

    function c_fortonnx_session_create_cpu_with_custom_ops( &
        model_path, context, custom_op_library) &
        bind(C, name='fortonnx_session_create_cpu_with_custom_ops') result(session)
      import :: c_char, c_ptr
      character(kind=c_char), intent(in) :: model_path(*)
      type(c_ptr), value :: context
      character(kind=c_char), intent(in) :: custom_op_library(*)
      type(c_ptr) :: session
    end function c_fortonnx_session_create_cpu_with_custom_ops

    function c_fortonnx_session_create_gpu(model_path, backend, device_id, &
        user_stream, use_tf32, trt_fp16, trt_cache, cache_path) &
        bind(C, name='fortonnx_session_create_gpu') result(session)
      import :: c_char, c_int, c_ptr
      character(kind=c_char), intent(in) :: model_path(*)
      integer(c_int), value :: backend, device_id
      type(c_ptr), value :: user_stream
      integer(c_int), value :: use_tf32, trt_fp16, trt_cache
      character(kind=c_char), intent(in) :: cache_path(*)
      type(c_ptr) :: session
    end function c_fortonnx_session_create_gpu

    function c_fortonnx_session_create_gpu_with_custom_ops(model_path, backend, &
        device_id, user_stream, use_tf32, trt_fp16, trt_cache, cache_path, &
        custom_op_library) bind(C, name='fortonnx_session_create_gpu_with_custom_ops') &
        result(session)
      import :: c_char, c_int, c_ptr
      character(kind=c_char), intent(in) :: model_path(*)
      integer(c_int), value :: backend, device_id
      type(c_ptr), value :: user_stream
      integer(c_int), value :: use_tf32, trt_fp16, trt_cache
      character(kind=c_char), intent(in) :: cache_path(*), custom_op_library(*)
      type(c_ptr) :: session
    end function c_fortonnx_session_create_gpu_with_custom_ops

    function c_fortonnx_session_get_features(session, input_features, &
        output_features) bind(C, name='fortonnx_session_get_features') result(code)
      import :: c_int, c_int64_t, c_ptr
      type(c_ptr), value :: session
      integer(c_int64_t), intent(out) :: input_features, output_features
      integer(c_int) :: code
    end function c_fortonnx_session_get_features

    function c_fortonnx_session_get_tensor_count(session, is_input, count) &
        bind(C, name='fortonnx_session_get_tensor_count') result(code)
      import :: c_int, c_int64_t, c_ptr
      type(c_ptr), value :: session
      integer(c_int), value :: is_input
      integer(c_int64_t), intent(out) :: count
      integer(c_int) :: code
    end function c_fortonnx_session_get_tensor_count

    function c_fortonnx_session_get_tensor_rank(session, is_input, rank) &
        bind(C, name='fortonnx_session_get_tensor_rank') result(code)
      import :: c_int, c_int64_t, c_ptr
      type(c_ptr), value :: session
      integer(c_int), value :: is_input
      integer(c_int64_t), intent(out) :: rank
      integer(c_int) :: code
    end function c_fortonnx_session_get_tensor_rank

    function c_fortonnx_session_get_tensor_shape(session, is_input, shape, rank) &
        bind(C, name='fortonnx_session_get_tensor_shape') result(code)
      import :: c_int, c_int64_t, c_ptr
      type(c_ptr), value :: session
      integer(c_int), value :: is_input
      integer(c_int64_t), intent(out) :: shape(*)
      integer(c_int64_t), value :: rank
      integer(c_int) :: code
    end function c_fortonnx_session_get_tensor_shape

    function c_fortonnx_session_get_tensor_rank_at(session, is_input, index, rank) &
        bind(C, name='fortonnx_session_get_tensor_rank_at') result(code)
      import :: c_int, c_int64_t, c_ptr
      type(c_ptr), value :: session
      integer(c_int), value :: is_input
      integer(c_int64_t), value :: index
      integer(c_int64_t), intent(out) :: rank
      integer(c_int) :: code
    end function c_fortonnx_session_get_tensor_rank_at

    function c_fortonnx_session_get_tensor_shape_at(session, is_input, index, &
        shape, rank) bind(C, name='fortonnx_session_get_tensor_shape_at') result(code)
      import :: c_int, c_int64_t, c_ptr
      type(c_ptr), value :: session
      integer(c_int), value :: is_input
      integer(c_int64_t), value :: index
      integer(c_int64_t), intent(out) :: shape(*)
      integer(c_int64_t), value :: rank
      integer(c_int) :: code
    end function c_fortonnx_session_get_tensor_shape_at

    function c_fortonnx_session_find_tensor(session, is_input, name, index) &
        bind(C, name='fortonnx_session_find_tensor') result(code)
      import :: c_char, c_int, c_int64_t, c_ptr
      type(c_ptr), value :: session
      integer(c_int), value :: is_input
      character(kind=c_char), intent(in) :: name(*)
      integer(c_int64_t), intent(out) :: index
      integer(c_int) :: code
    end function c_fortonnx_session_find_tensor

    function c_fortonnx_session_begin_bind(session) &
        bind(C, name='fortonnx_session_begin_bind') result(code)
      import :: c_int, c_ptr
      type(c_ptr), value :: session
      integer(c_int) :: code
    end function c_fortonnx_session_begin_bind

    function c_fortonnx_session_bind_input(session, index, data, memory_backend, &
        shape, rank) bind(C, name='fortonnx_session_bind_input') result(code)
      import :: c_int, c_int64_t, c_ptr
      type(c_ptr), value :: session, data
      integer(c_int64_t), value :: index
      integer(c_int), value :: memory_backend
      integer(c_int64_t), intent(in) :: shape(*)
      integer(c_int64_t), value :: rank
      integer(c_int) :: code
    end function c_fortonnx_session_bind_input

    function c_fortonnx_session_bind_input_typed(session, index, data, memory_backend, &
        shape, rank, element_type) bind(C, name='fortonnx_session_bind_input_typed') result(code)
      import :: c_int, c_int64_t, c_ptr
      type(c_ptr), value :: session, data
      integer(c_int64_t), value :: index
      integer(c_int), value :: memory_backend, element_type
      integer(c_int64_t), intent(in) :: shape(*)
      integer(c_int64_t), value :: rank
      integer(c_int) :: code
    end function c_fortonnx_session_bind_input_typed

    function c_fortonnx_session_bind_output(session, index, data, memory_backend, &
        shape, rank) bind(C, name='fortonnx_session_bind_output') result(code)
      import :: c_int, c_int64_t, c_ptr
      type(c_ptr), value :: session, data
      integer(c_int64_t), value :: index
      integer(c_int), value :: memory_backend
      integer(c_int64_t), intent(in) :: shape(*)
      integer(c_int64_t), value :: rank
      integer(c_int) :: code
    end function c_fortonnx_session_bind_output

    function c_fortonnx_session_bind_output_typed(session, index, data, memory_backend, &
        shape, rank, element_type) bind(C, name='fortonnx_session_bind_output_typed') result(code)
      import :: c_int, c_int64_t, c_ptr
      type(c_ptr), value :: session, data
      integer(c_int64_t), value :: index
      integer(c_int), value :: memory_backend, element_type
      integer(c_int64_t), intent(in) :: shape(*)
      integer(c_int64_t), value :: rank
      integer(c_int) :: code
    end function c_fortonnx_session_bind_output_typed

    function c_fortonnx_session_bind(session, input, output, memory_backend, &
        batch_size, input_features, output_features) &
        bind(C, name='fortonnx_session_bind') result(code)
      import :: c_int, c_int64_t, c_ptr
      type(c_ptr), value :: session, input, output
      integer(c_int), value :: memory_backend
      integer(c_int64_t), value :: batch_size, input_features, output_features
      integer(c_int) :: code
    end function c_fortonnx_session_bind

    function c_fortonnx_session_bind_tensor(session, input, output, &
        memory_backend, input_shape, input_rank, output_shape, output_rank) &
        bind(C, name='fortonnx_session_bind_tensor') result(code)
      import :: c_int, c_int64_t, c_ptr
      type(c_ptr), value :: session, input, output
      integer(c_int), value :: memory_backend
      integer(c_int64_t), intent(in) :: input_shape(*), output_shape(*)
      integer(c_int64_t), value :: input_rank, output_rank
      integer(c_int) :: code
    end function c_fortonnx_session_bind_tensor

    function c_fortonnx_session_bind_tensor_typed(session, input, output, &
        memory_backend, input_shape, input_rank, output_shape, output_rank, input_type, output_type) &
        bind(C, name='fortonnx_session_bind_tensor_typed') result(code)
      import :: c_int, c_int64_t, c_ptr
      type(c_ptr), value :: session, input, output
      integer(c_int), value :: memory_backend, input_type, output_type
      integer(c_int64_t), intent(in) :: input_shape(*), output_shape(*)
      integer(c_int64_t), value :: input_rank, output_rank
      integer(c_int) :: code
    end function c_fortonnx_session_bind_tensor_typed

    function c_fortonnx_session_run(session) &
        bind(C, name='fortonnx_session_run') result(code)
      import :: c_int, c_ptr
      type(c_ptr), value :: session
      integer(c_int) :: code
    end function c_fortonnx_session_run

    subroutine c_fortonnx_session_destroy(session) &
        bind(C, name='fortonnx_session_destroy')
      import :: c_ptr
      type(c_ptr), value :: session
    end subroutine c_fortonnx_session_destroy

    function c_fortonnx_last_error(session, buffer, buffer_size) &
        bind(C, name='fortonnx_last_error') result(length)
      import :: c_char, c_ptr, c_size_t
      type(c_ptr), value :: session
      character(kind=c_char), intent(out) :: buffer(*)
      integer(c_size_t), value :: buffer_size
      integer(c_size_t) :: length
    end function c_fortonnx_last_error
  end interface
end module fortonnx_c_api
