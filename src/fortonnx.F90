module fortonnx
  use, intrinsic :: iso_c_binding
  use fortonnx_c_api
#ifdef FORTONNX_WITH_CUDA_FORTRAN
  use cudafor
#endif
  implicit none
  private

  integer(c_int), parameter, public :: fortonnx_cpu = 0_c_int
  integer(c_int), parameter, public :: fortonnx_cuda = 1_c_int
  integer(c_int), parameter, public :: fortonnx_tensorrt = 2_c_int
  integer(c_int), parameter, public :: fortonnx_output = 0_c_int
  integer(c_int), parameter, public :: fortonnx_input = 1_c_int
  integer(c_int), parameter, public :: fortonnx_success = 0_c_int
  integer(c_int), parameter, public :: fortonnx_invalid_argument = 1_c_int
  integer(c_int), parameter, public :: fortonnx_runtime_error = 2_c_int

  integer(c_int), parameter, public :: fortonnx_float32 = 1_c_int
  integer(c_int), parameter, public :: fortonnx_bool = 9_c_int
  integer(c_int), parameter, public :: fortonnx_float64 = 11_c_int

  type, public :: fortonnx_options
    integer(c_int) :: backend = fortonnx_cpu
    integer(c_int) :: device_id = 0_c_int
    integer(c_int) :: intra_op_threads = 0_c_int
    integer(c_int) :: inter_op_threads = 1_c_int
    logical :: use_tf32 = .false.
    logical :: tensorrt_fp16 = .false.
    logical :: tensorrt_engine_cache = .false.
    character(len=:), allocatable :: tensorrt_cache_path
    type(c_ptr) :: user_compute_stream = c_null_ptr
  end type fortonnx_options

  type, public :: fortonnx_status
    integer(c_int) :: code = fortonnx_success
    character(len=:), allocatable :: message
  contains
    procedure, public :: ok => status_ok
    procedure, public :: clear => status_clear
  end type fortonnx_status

  type, public :: fortonnx_tensor
    private
    type(c_ptr) :: data = c_null_ptr
    integer(c_int64_t), allocatable :: shape(:)
    character(len=:), allocatable :: name
    integer(c_int) :: memory_backend = -1_c_int
    integer(c_int) :: element_type = fortonnx_float32
  end type fortonnx_tensor

  interface fortonnx_host_tensor
    module procedure make_host_tensor_float32, make_host_tensor_float64, make_host_tensor_bool
  end interface fortonnx_host_tensor
  public :: fortonnx_host_tensor
  public :: fortonnx_pointer_tensor

#ifdef FORTONNX_WITH_CUDA_FORTRAN
  interface fortonnx_device_tensor
    module procedure make_device_tensor_1d
    module procedure make_device_tensor_1d_bool
    module procedure make_device_tensor_1d_float64
    module procedure make_device_tensor_2d
    module procedure make_device_tensor_2d_bool
    module procedure make_device_tensor_2d_float64
    module procedure make_device_tensor_3d
    module procedure make_device_tensor_3d_bool
    module procedure make_device_tensor_3d_float64
    module procedure make_device_tensor_4d
    module procedure make_device_tensor_4d_bool
    module procedure make_device_tensor_4d_float64
    module procedure make_device_tensor_5d
    module procedure make_device_tensor_5d_bool
    module procedure make_device_tensor_5d_float64
    module procedure make_device_tensor_6d
    module procedure make_device_tensor_6d_bool
    module procedure make_device_tensor_6d_float64
    module procedure make_device_tensor_7d
    module procedure make_device_tensor_7d_bool
    module procedure make_device_tensor_7d_float64
  end interface fortonnx_device_tensor
  public :: fortonnx_device_tensor
#endif

  type :: tensor_metadata
    integer(c_int64_t), allocatable :: shape(:)
  end type tensor_metadata

  type :: model_slot
    type(c_ptr) :: session = c_null_ptr
    character(len=:), allocatable :: name
    character(len=:), allocatable :: path
    type(tensor_metadata), allocatable :: inputs(:)
    type(tensor_metadata), allocatable :: outputs(:)
    integer(c_int64_t) :: batch_size = 0_c_int64_t
    logical :: bound = .false.
  end type model_slot

#ifdef FORTONNX_WITH_CUDA_FORTRAN
  type, public :: fortonnx_cuda_stream
    private
    integer(kind=cuda_stream_kind) :: handle = 0
    logical :: active = .false.
  contains
    procedure, public :: create => stream_create
    procedure, public :: synchronize => stream_synchronize
    procedure, public :: c_pointer => stream_c_pointer
    procedure, public :: close => stream_close
    final :: stream_finalize
  end type fortonnx_cuda_stream
#endif

  type, public :: fortonnx_runtime
    private
    type(model_slot), allocatable :: models(:)
    type(c_ptr) :: cpu_context = c_null_ptr
    type(fortonnx_options) :: options
    logical :: initialized = .false.
  contains
    procedure, public :: initialize => runtime_initialize
    procedure, public :: load => runtime_load
    procedure, private :: bind_host_array => runtime_bind_host_array
    procedure, private :: bind_host_bool_bool => runtime_bind_host_bool_bool
    procedure, private :: bind_host_bool_float64 => runtime_bind_host_bool_float64
    procedure, private :: bind_host_bool_float32 => runtime_bind_host_bool_float32
    procedure, private :: bind_host_float64_bool => runtime_bind_host_float64_bool
    procedure, private :: bind_host_float64_float64 => runtime_bind_host_float64_float64
    procedure, private :: bind_host_float64_float32 => runtime_bind_host_float64_float32
    procedure, private :: bind_host_float32_bool => runtime_bind_host_float32_bool
    procedure, private :: bind_host_float32_float64 => runtime_bind_host_float32_float64
    procedure, private :: bind_pointer_batch => runtime_bind_pointer_batch
    procedure, private :: bind_pointer_shapes => runtime_bind_pointer_shapes
    procedure, private :: bind_tensor_pair => runtime_bind_tensor_pair
    procedure, private :: bind_tensor_inputs => runtime_bind_tensor_inputs
    procedure, private :: bind_tensor_arrays => runtime_bind_tensor_arrays
#ifdef FORTONNX_WITH_CUDA_FORTRAN
    procedure, private :: bind_device_array_1d => runtime_bind_device_array_1d
    procedure, private :: bind_device_array_2d => runtime_bind_device_array_2d
    procedure, private :: bind_device_array_3d => runtime_bind_device_array_3d
    procedure, private :: bind_device_array_4d => runtime_bind_device_array_4d
    procedure, private :: bind_device_array_5d => runtime_bind_device_array_5d
    procedure, private :: bind_device_array_6d => runtime_bind_device_array_6d
    procedure, private :: bind_device_array_7d => runtime_bind_device_array_7d
    procedure, private :: bind_device_7d_bool_bool => runtime_bind_device_7d_bool_bool
    procedure, private :: bind_device_7d_bool_float64 => runtime_bind_device_7d_bool_float64
    procedure, private :: bind_device_7d_bool_float32 => runtime_bind_device_7d_bool_float32
    procedure, private :: bind_device_7d_float64_bool => runtime_bind_device_7d_float64_bool
    procedure, private :: bind_device_7d_float64_float64 => runtime_bind_device_7d_float64_float64
    procedure, private :: bind_device_7d_float64_float32 => runtime_bind_device_7d_float64_float32
    procedure, private :: bind_device_7d_float32_bool => runtime_bind_device_7d_float32_bool
    procedure, private :: bind_device_7d_float32_float64 => runtime_bind_device_7d_float32_float64
    procedure, private :: bind_device_6d_bool_bool => runtime_bind_device_6d_bool_bool
    procedure, private :: bind_device_6d_bool_float64 => runtime_bind_device_6d_bool_float64
    procedure, private :: bind_device_6d_bool_float32 => runtime_bind_device_6d_bool_float32
    procedure, private :: bind_device_6d_float64_bool => runtime_bind_device_6d_float64_bool
    procedure, private :: bind_device_6d_float64_float64 => runtime_bind_device_6d_float64_float64
    procedure, private :: bind_device_6d_float64_float32 => runtime_bind_device_6d_float64_float32
    procedure, private :: bind_device_6d_float32_bool => runtime_bind_device_6d_float32_bool
    procedure, private :: bind_device_6d_float32_float64 => runtime_bind_device_6d_float32_float64
    procedure, private :: bind_device_5d_bool_bool => runtime_bind_device_5d_bool_bool
    procedure, private :: bind_device_5d_bool_float64 => runtime_bind_device_5d_bool_float64
    procedure, private :: bind_device_5d_bool_float32 => runtime_bind_device_5d_bool_float32
    procedure, private :: bind_device_5d_float64_bool => runtime_bind_device_5d_float64_bool
    procedure, private :: bind_device_5d_float64_float64 => runtime_bind_device_5d_float64_float64
    procedure, private :: bind_device_5d_float64_float32 => runtime_bind_device_5d_float64_float32
    procedure, private :: bind_device_5d_float32_bool => runtime_bind_device_5d_float32_bool
    procedure, private :: bind_device_5d_float32_float64 => runtime_bind_device_5d_float32_float64
    procedure, private :: bind_device_4d_bool_bool => runtime_bind_device_4d_bool_bool
    procedure, private :: bind_device_4d_bool_float64 => runtime_bind_device_4d_bool_float64
    procedure, private :: bind_device_4d_bool_float32 => runtime_bind_device_4d_bool_float32
    procedure, private :: bind_device_4d_float64_bool => runtime_bind_device_4d_float64_bool
    procedure, private :: bind_device_4d_float64_float64 => runtime_bind_device_4d_float64_float64
    procedure, private :: bind_device_4d_float64_float32 => runtime_bind_device_4d_float64_float32
    procedure, private :: bind_device_4d_float32_bool => runtime_bind_device_4d_float32_bool
    procedure, private :: bind_device_4d_float32_float64 => runtime_bind_device_4d_float32_float64
    procedure, private :: bind_device_3d_bool_bool => runtime_bind_device_3d_bool_bool
    procedure, private :: bind_device_3d_bool_float64 => runtime_bind_device_3d_bool_float64
    procedure, private :: bind_device_3d_bool_float32 => runtime_bind_device_3d_bool_float32
    procedure, private :: bind_device_3d_float64_bool => runtime_bind_device_3d_float64_bool
    procedure, private :: bind_device_3d_float64_float64 => runtime_bind_device_3d_float64_float64
    procedure, private :: bind_device_3d_float64_float32 => runtime_bind_device_3d_float64_float32
    procedure, private :: bind_device_3d_float32_bool => runtime_bind_device_3d_float32_bool
    procedure, private :: bind_device_3d_float32_float64 => runtime_bind_device_3d_float32_float64
    procedure, private :: bind_device_2d_bool_bool => runtime_bind_device_2d_bool_bool
    procedure, private :: bind_device_2d_bool_float64 => runtime_bind_device_2d_bool_float64
    procedure, private :: bind_device_2d_bool_float32 => runtime_bind_device_2d_bool_float32
    procedure, private :: bind_device_2d_float64_bool => runtime_bind_device_2d_float64_bool
    procedure, private :: bind_device_2d_float64_float64 => runtime_bind_device_2d_float64_float64
    procedure, private :: bind_device_2d_float64_float32 => runtime_bind_device_2d_float64_float32
    procedure, private :: bind_device_2d_float32_bool => runtime_bind_device_2d_float32_bool
    procedure, private :: bind_device_2d_float32_float64 => runtime_bind_device_2d_float32_float64
    procedure, private :: bind_device_1d_bool_bool => runtime_bind_device_1d_bool_bool
    procedure, private :: bind_device_1d_bool_float64 => runtime_bind_device_1d_bool_float64
    procedure, private :: bind_device_1d_bool_float32 => runtime_bind_device_1d_bool_float32
    procedure, private :: bind_device_1d_float64_bool => runtime_bind_device_1d_float64_bool
    procedure, private :: bind_device_1d_float64_float64 => runtime_bind_device_1d_float64_float64
    procedure, private :: bind_device_1d_float64_float32 => runtime_bind_device_1d_float64_float32
    procedure, private :: bind_device_1d_float32_bool => runtime_bind_device_1d_float32_bool
    procedure, private :: bind_device_1d_float32_float64 => runtime_bind_device_1d_float32_float64
    generic, public :: bind => bind_device_1d_float32_float64
    generic, public :: bind => bind_device_1d_float32_bool
    generic, public :: bind => bind_device_1d_float64_float32
    generic, public :: bind => bind_device_1d_float64_float64
    generic, public :: bind => bind_device_1d_float64_bool
    generic, public :: bind => bind_device_1d_bool_float32
    generic, public :: bind => bind_device_1d_bool_float64
    generic, public :: bind => bind_device_1d_bool_bool
    generic, public :: bind => bind_device_2d_float32_float64
    generic, public :: bind => bind_device_2d_float32_bool
    generic, public :: bind => bind_device_2d_float64_float32
    generic, public :: bind => bind_device_2d_float64_float64
    generic, public :: bind => bind_device_2d_float64_bool
    generic, public :: bind => bind_device_2d_bool_float32
    generic, public :: bind => bind_device_2d_bool_float64
    generic, public :: bind => bind_device_2d_bool_bool
    generic, public :: bind => bind_device_3d_float32_float64
    generic, public :: bind => bind_device_3d_float32_bool
    generic, public :: bind => bind_device_3d_float64_float32
    generic, public :: bind => bind_device_3d_float64_float64
    generic, public :: bind => bind_device_3d_float64_bool
    generic, public :: bind => bind_device_3d_bool_float32
    generic, public :: bind => bind_device_3d_bool_float64
    generic, public :: bind => bind_device_3d_bool_bool
    generic, public :: bind => bind_device_4d_float32_float64
    generic, public :: bind => bind_device_4d_float32_bool
    generic, public :: bind => bind_device_4d_float64_float32
    generic, public :: bind => bind_device_4d_float64_float64
    generic, public :: bind => bind_device_4d_float64_bool
    generic, public :: bind => bind_device_4d_bool_float32
    generic, public :: bind => bind_device_4d_bool_float64
    generic, public :: bind => bind_device_4d_bool_bool
    generic, public :: bind => bind_device_5d_float32_float64
    generic, public :: bind => bind_device_5d_float32_bool
    generic, public :: bind => bind_device_5d_float64_float32
    generic, public :: bind => bind_device_5d_float64_float64
    generic, public :: bind => bind_device_5d_float64_bool
    generic, public :: bind => bind_device_5d_bool_float32
    generic, public :: bind => bind_device_5d_bool_float64
    generic, public :: bind => bind_device_5d_bool_bool
    generic, public :: bind => bind_device_6d_float32_float64
    generic, public :: bind => bind_device_6d_float32_bool
    generic, public :: bind => bind_device_6d_float64_float32
    generic, public :: bind => bind_device_6d_float64_float64
    generic, public :: bind => bind_device_6d_float64_bool
    generic, public :: bind => bind_device_6d_bool_float32
    generic, public :: bind => bind_device_6d_bool_float64
    generic, public :: bind => bind_device_6d_bool_bool
    generic, public :: bind => bind_device_7d_float32_float64
    generic, public :: bind => bind_device_7d_float32_bool
    generic, public :: bind => bind_device_7d_float64_float32
    generic, public :: bind => bind_device_7d_float64_float64
    generic, public :: bind => bind_device_7d_float64_bool
    generic, public :: bind => bind_device_7d_bool_float32
    generic, public :: bind => bind_device_7d_bool_float64
    generic, public :: bind => bind_device_7d_bool_bool
    generic, public :: bind => bind_host_array, bind_pointer_batch, &
        bind_pointer_shapes, bind_tensor_pair, bind_tensor_inputs, &
        bind_tensor_arrays, bind_device_array_1d, &
        bind_device_array_2d, bind_device_array_3d, bind_device_array_4d, &
        bind_device_array_5d, bind_device_array_6d, bind_device_array_7d
#else
    generic, public :: bind => bind_host_array, bind_pointer_batch, bind_pointer_shapes, &
        bind_tensor_pair, bind_tensor_inputs, bind_tensor_arrays
#endif
    generic, public :: bind => bind_host_float32_float64
    generic, public :: bind => bind_host_float32_bool
    generic, public :: bind => bind_host_float64_float32
    generic, public :: bind => bind_host_float64_float64
    generic, public :: bind => bind_host_float64_bool
    generic, public :: bind => bind_host_bool_float32
    generic, public :: bind => bind_host_bool_float64
    generic, public :: bind => bind_host_bool_bool
    procedure, public :: run => runtime_run
    procedure, public :: run_all => runtime_run_all
    procedure, public :: features => runtime_features
    procedure, public :: shapes => runtime_shapes
    procedure, public :: io_counts => runtime_io_counts
    procedure, public :: tensor_shape => runtime_tensor_shape
    procedure, public :: tensor_type => runtime_tensor_type
    procedure, public :: find => runtime_find
    procedure, public :: size => runtime_size
    procedure, public :: is_initialized => runtime_is_initialized
    procedure, public :: close => runtime_close
    final :: runtime_finalize
  end type fortonnx_runtime

contains

  function host_data_pointer(array) result(data)
    type(*), contiguous, target, intent(inout) :: array(..)
    type(c_ptr) :: data

    ! The assumed-type descriptor avoids nvfortran 25.9's incorrect
    ! C_LOC base-address calculation for typed assumed-rank arrays.
    data = c_loc(array)
  end function host_data_pointer

  function make_host_tensor_float32(array, name) result(tensor)
    real(c_float), contiguous, target, intent(inout) :: array(..)
    character(len=*), intent(in), optional :: name
    type(fortonnx_tensor) :: tensor
    integer(c_int64_t), allocatable :: fortran_shape(:)

    tensor%element_type = fortonnx_float32
    tensor%data = host_data_pointer(array)
    fortran_shape = shape(array, kind=c_int64_t)
    allocate(tensor%shape(size(fortran_shape)))
    ! Avoid allocatable reallocation: nvfortran 25.9 miscomputes the
    ! reverse-section extent for ranks greater than one in this assignment.
    tensor%shape(:) = fortran_shape(size(fortran_shape):1:-1)
    tensor%memory_backend = fortonnx_cpu
    if (present(name)) tensor%name = trim(name)
  end function make_host_tensor_float32

  function make_host_tensor_float64(array, name) result(tensor)
    real(c_double), contiguous, target, intent(inout) :: array(..)
    character(len=*), intent(in), optional :: name
    type(fortonnx_tensor) :: tensor
    integer(c_int64_t), allocatable :: fortran_shape(:)

    tensor%element_type = fortonnx_float64
    tensor%data = host_data_pointer(array)
    fortran_shape = shape(array, kind=c_int64_t)
    allocate(tensor%shape(size(fortran_shape)))
    tensor%shape(:) = fortran_shape(size(fortran_shape):1:-1)
    tensor%memory_backend = fortonnx_cpu
    if (present(name)) tensor%name = trim(name)
  end function make_host_tensor_float64

  function make_host_tensor_bool(array, name) result(tensor)
    logical(c_bool), contiguous, target, intent(inout) :: array(..)
    character(len=*), intent(in), optional :: name
    type(fortonnx_tensor) :: tensor
    integer(c_int64_t), allocatable :: fortran_shape(:)

    tensor%element_type = fortonnx_bool
    tensor%data = host_data_pointer(array)
    fortran_shape = shape(array, kind=c_int64_t)
    allocate(tensor%shape(size(fortran_shape)))
    tensor%shape(:) = fortran_shape(size(fortran_shape):1:-1)
    tensor%memory_backend = fortonnx_cpu
    if (present(name)) tensor%name = trim(name)
  end function make_host_tensor_bool

  function fortonnx_pointer_tensor(data, onnx_shape, memory_backend, name, element_type) result(tensor)
    type(c_ptr), intent(in) :: data
    integer(c_int64_t), intent(in) :: onnx_shape(:)
    integer(c_int), intent(in) :: memory_backend
    character(len=*), intent(in), optional :: name
    integer(c_int), intent(in), optional :: element_type
    type(fortonnx_tensor) :: tensor

    if (present(element_type)) tensor%element_type = element_type
    tensor%data = data
    allocate(tensor%shape(size(onnx_shape)))
    tensor%shape = onnx_shape
    tensor%memory_backend = memory_backend
    if (present(name)) tensor%name = trim(name)
  end function fortonnx_pointer_tensor

  logical function status_ok(self)
    class(fortonnx_status), intent(in) :: self
    status_ok = self%code == fortonnx_success
  end function status_ok

  subroutine status_clear(self)
    class(fortonnx_status), intent(inout) :: self
    self%code = fortonnx_success
    self%message = ''
  end subroutine status_clear

  subroutine set_status(status, code, message)
    type(fortonnx_status), intent(inout) :: status
    integer(c_int), intent(in) :: code
    character(len=*), intent(in) :: message
    status%code = code
    status%message = trim(message)
  end subroutine set_status

  function runtime_error_message(session) result(message)
    type(c_ptr), intent(in) :: session
    character(len=:), allocatable :: message
    character(kind=c_char) :: buffer(2048)
    integer(c_size_t) :: ignored
    integer :: index, length

    buffer = c_null_char
    ignored = c_fortonnx_last_error(session, buffer, size(buffer, kind=c_size_t))
    length = 0
    do index = 1, size(buffer)
      if (buffer(index) == c_null_char) exit
      length = length + 1
    end do
    allocate(character(len=length) :: message)
    do index = 1, length
      message(index:index) = achar(iachar(buffer(index)))
    end do
  end function runtime_error_message

  subroutine set_c_error(status, context, session)
    type(fortonnx_status), intent(inout) :: status
    character(len=*), intent(in) :: context
    type(c_ptr), intent(in) :: session
    character(len=:), allocatable :: detail

    detail = runtime_error_message(session)
    call set_status(status, fortonnx_runtime_error, trim(context) // ': ' // detail)
  end subroutine set_c_error

  subroutine runtime_initialize(self, options, status)
    class(fortonnx_runtime), intent(inout) :: self
    type(fortonnx_options), intent(in) :: options
    type(fortonnx_status), intent(inout) :: status

    call status%clear()
    call self%close()
    if (options%backend < fortonnx_cpu .or. options%backend > fortonnx_tensorrt) then
      call set_status(status, fortonnx_invalid_argument, 'unknown inference backend')
      return
    end if
    if (options%intra_op_threads < 0 .or. options%inter_op_threads < 1) then
      call set_status(status, fortonnx_invalid_argument, &
          'thread counts must satisfy intra_op >= 0 and inter_op >= 1')
      return
    end if
    if (options%tensorrt_engine_cache) then
      if (.not. allocated(options%tensorrt_cache_path)) then
        call set_status(status, fortonnx_invalid_argument, &
            'TensorRT cache is enabled but no cache path was supplied')
        return
      end if
      if (len_trim(options%tensorrt_cache_path) == 0) then
        call set_status(status, fortonnx_invalid_argument, &
            'TensorRT cache path must not be empty')
        return
      end if
    end if

    self%options = options
    if (options%backend == fortonnx_cpu) then
      self%cpu_context = c_fortonnx_cpu_context_create( &
          options%intra_op_threads, options%inter_op_threads)
      if (.not. c_associated(self%cpu_context)) then
        call set_c_error(status, 'failed to initialize the CPU runtime', c_null_ptr)
        return
      end if
    end if
    allocate(self%models(0))
    self%initialized = .true.
  end subroutine runtime_initialize

  subroutine runtime_load(self, name, model_path, status, custom_op_library)
    class(fortonnx_runtime), intent(inout) :: self
    character(len=*), intent(in) :: name, model_path
    type(fortonnx_status), intent(inout) :: status
    character(len=*), intent(in), optional :: custom_op_library
    type(model_slot), allocatable :: expanded(:)
    type(model_slot) :: candidate
    character(kind=c_char, len=:), allocatable :: c_model_path, c_cache_path, &
        c_custom_op_library
    integer(c_int) :: code
    integer(c_int64_t) :: tensor_rank, input_count, output_count
    integer :: old_size, tensor_index

    call status%clear()
    if (.not. self%initialized) then
      call set_status(status, fortonnx_invalid_argument, 'runtime is not initialized')
      return
    end if
    if (len_trim(name) == 0 .or. len_trim(model_path) == 0) then
      call set_status(status, fortonnx_invalid_argument, 'model name and path must not be empty')
      return
    end if
    if (self%find(name) /= 0) then
      call set_status(status, fortonnx_invalid_argument, 'duplicate model name: ' // trim(name))
      return
    end if

    c_model_path = trim(model_path) // c_null_char
    if (present(custom_op_library)) then
      if (len_trim(custom_op_library) == 0) then
        call set_status(status, fortonnx_invalid_argument, &
            'custom-op library path must not be empty')
        return
      end if
      c_custom_op_library = trim(custom_op_library) // c_null_char
    else
      c_custom_op_library = c_null_char
    end if
    if (self%options%backend == fortonnx_cpu) then
      candidate%session = c_fortonnx_session_create_cpu_with_custom_ops( &
          c_model_path, self%cpu_context, c_custom_op_library)
    else
      if (allocated(self%options%tensorrt_cache_path)) then
        c_cache_path = trim(self%options%tensorrt_cache_path) // c_null_char
      else
        c_cache_path = c_null_char
      end if
      candidate%session = c_fortonnx_session_create_gpu_with_custom_ops( &
          c_model_path, self%options%backend, self%options%device_id, &
          self%options%user_compute_stream, &
          merge(1_c_int, 0_c_int, self%options%use_tf32), &
          merge(1_c_int, 0_c_int, self%options%tensorrt_fp16), &
          merge(1_c_int, 0_c_int, self%options%tensorrt_engine_cache), c_cache_path, &
          c_custom_op_library)
    end if
    if (.not. c_associated(candidate%session)) then
      call set_c_error(status, 'failed to load ' // trim(model_path), c_null_ptr)
      return
    end if
    code = c_fortonnx_session_get_tensor_count(candidate%session, fortonnx_input, input_count)
    if (code == 0_c_int) then
      code = c_fortonnx_session_get_tensor_count( &
          candidate%session, fortonnx_output, output_count)
    end if
    if (code == 0_c_int) then
      if (input_count < 1_c_int64_t .or. output_count < 1_c_int64_t .or. &
          input_count > int(huge(0), c_int64_t) .or. &
          output_count > int(huge(0), c_int64_t)) code = 1_c_int
    end if
    if (code == 0_c_int) then
      allocate(candidate%inputs(int(input_count)), candidate%outputs(int(output_count)))
      do tensor_index = 1, size(candidate%inputs)
        code = c_fortonnx_session_get_tensor_rank_at(candidate%session, fortonnx_input, &
            int(tensor_index - 1, c_int64_t), tensor_rank)
        if (code /= 0_c_int) exit
        allocate(candidate%inputs(tensor_index)%shape(int(tensor_rank)))
        code = c_fortonnx_session_get_tensor_shape_at(candidate%session, fortonnx_input, &
            int(tensor_index - 1, c_int64_t), candidate%inputs(tensor_index)%shape, tensor_rank)
        if (code /= 0_c_int) exit
      end do
    end if
    if (code == 0_c_int) then
      do tensor_index = 1, size(candidate%outputs)
        code = c_fortonnx_session_get_tensor_rank_at(candidate%session, fortonnx_output, &
            int(tensor_index - 1, c_int64_t), tensor_rank)
        if (code /= 0_c_int) exit
        allocate(candidate%outputs(tensor_index)%shape(int(tensor_rank)))
        code = c_fortonnx_session_get_tensor_shape_at(candidate%session, fortonnx_output, &
            int(tensor_index - 1, c_int64_t), candidate%outputs(tensor_index)%shape, tensor_rank)
        if (code /= 0_c_int) exit
      end do
    end if
    if (code /= 0_c_int) then
      call set_c_error(status, 'failed to inspect ' // trim(model_path), &
          candidate%session)
      call c_fortonnx_session_destroy(candidate%session)
      candidate%session = c_null_ptr
      return
    end if
    candidate%name = trim(name)
    candidate%path = trim(model_path)
    old_size = size(self%models)
    allocate(expanded(old_size + 1))
    if (old_size > 0) expanded(1:old_size) = self%models
    expanded(old_size + 1) = candidate
    call move_alloc(expanded, self%models)
  end subroutine runtime_load

  subroutine reverse_shape(array, onnx_shape)
    class(*), contiguous, intent(in) :: array(..)
    integer(c_int64_t), allocatable, intent(out) :: onnx_shape(:)
    integer(c_int64_t), allocatable :: fortran_shape(:)

    fortran_shape = shape(array, kind=c_int64_t)
    allocate(onnx_shape(size(fortran_shape)))
    onnx_shape(:) = fortran_shape(size(fortran_shape):1:-1)
  end subroutine reverse_shape

  subroutine apply_legacy_vector_shape(self, name, input_shape, output_shape)
    class(fortonnx_runtime), intent(in) :: self
    character(len=*), intent(in) :: name
    integer(c_int64_t), allocatable, intent(inout) :: input_shape(:), output_shape(:)
    integer :: index

    index = self%find(name)
    if (index == 0) return
    if (size(self%models(index)%inputs) == 1 .and. size(input_shape) == 1 .and. &
        size(self%models(index)%inputs(1)%shape) == 2) then
      if (self%models(index)%inputs(1)%shape(2) == 1_c_int64_t) then
        input_shape = [input_shape(1), 1_c_int64_t]
      end if
    end if
    if (size(self%models(index)%outputs) == 1 .and. size(output_shape) == 1 .and. &
        size(self%models(index)%outputs(1)%shape) == 2) then
      if (self%models(index)%outputs(1)%shape(2) == 1_c_int64_t) then
        output_shape = [output_shape(1), 1_c_int64_t]
      end if
    end if
  end subroutine apply_legacy_vector_shape

  subroutine runtime_bind_host_array(self, name, input, output, status)
    class(fortonnx_runtime), intent(inout) :: self
    character(len=*), intent(in) :: name
    real(c_float), contiguous, target, intent(inout) :: input(..)
    real(c_float), contiguous, target, intent(inout) :: output(..)
    type(fortonnx_status), intent(inout) :: status
    integer(c_int64_t), allocatable :: input_shape(:), output_shape(:)

    call status%clear()
    if (self%options%backend /= fortonnx_cpu) then
      call set_status(status, fortonnx_invalid_argument, 'host buffers require the CPU backend')
      return
    end if
    if (size(input) < 1 .or. size(output) < 1) then
      call set_status(status, fortonnx_invalid_argument, 'tensor arrays must not be empty')
      return
    end if
    call reverse_shape(input, input_shape)
    call reverse_shape(output, output_shape)
    call apply_legacy_vector_shape(self, name, input_shape, output_shape)
    call runtime_bind_pointer_shapes(self, name, host_data_pointer(input), host_data_pointer(output), &
        input_shape, output_shape, status, fortonnx_float32, fortonnx_float32)
  end subroutine runtime_bind_host_array

  subroutine runtime_bind_host_float32_float64(self, name, input, output, status)
    class(fortonnx_runtime), intent(inout) :: self
    character(len=*), intent(in) :: name
    real(c_float), contiguous, target, intent(inout) :: input(..)
    real(c_double), contiguous, target, intent(inout) :: output(..)
    type(fortonnx_status), intent(inout) :: status
    integer(c_int64_t), allocatable :: input_shape(:), output_shape(:)

    call status%clear()
    if (self%options%backend /= fortonnx_cpu) then
      call set_status(status, fortonnx_invalid_argument, 'host buffers require the CPU backend')
      return
    end if
    if (size(input) < 1 .or. size(output) < 1) then
      call set_status(status, fortonnx_invalid_argument, 'tensor arrays must not be empty')
      return
    end if
    call reverse_shape(input, input_shape)
    call reverse_shape(output, output_shape)
    call apply_legacy_vector_shape(self, name, input_shape, output_shape)
    call runtime_bind_pointer_shapes(self, name, host_data_pointer(input), host_data_pointer(output), &
        input_shape, output_shape, status, fortonnx_float32, fortonnx_float64)
  end subroutine runtime_bind_host_float32_float64

  subroutine runtime_bind_host_float32_bool(self, name, input, output, status)
    class(fortonnx_runtime), intent(inout) :: self
    character(len=*), intent(in) :: name
    real(c_float), contiguous, target, intent(inout) :: input(..)
    logical(c_bool), contiguous, target, intent(inout) :: output(..)
    type(fortonnx_status), intent(inout) :: status
    integer(c_int64_t), allocatable :: input_shape(:), output_shape(:)

    call status%clear()
    if (self%options%backend /= fortonnx_cpu) then
      call set_status(status, fortonnx_invalid_argument, 'host buffers require the CPU backend')
      return
    end if
    if (size(input) < 1 .or. size(output) < 1) then
      call set_status(status, fortonnx_invalid_argument, 'tensor arrays must not be empty')
      return
    end if
    call reverse_shape(input, input_shape)
    call reverse_shape(output, output_shape)
    call apply_legacy_vector_shape(self, name, input_shape, output_shape)
    call runtime_bind_pointer_shapes(self, name, host_data_pointer(input), host_data_pointer(output), &
        input_shape, output_shape, status, fortonnx_float32, fortonnx_bool)
  end subroutine runtime_bind_host_float32_bool

  subroutine runtime_bind_host_float64_float32(self, name, input, output, status)
    class(fortonnx_runtime), intent(inout) :: self
    character(len=*), intent(in) :: name
    real(c_double), contiguous, target, intent(inout) :: input(..)
    real(c_float), contiguous, target, intent(inout) :: output(..)
    type(fortonnx_status), intent(inout) :: status
    integer(c_int64_t), allocatable :: input_shape(:), output_shape(:)

    call status%clear()
    if (self%options%backend /= fortonnx_cpu) then
      call set_status(status, fortonnx_invalid_argument, 'host buffers require the CPU backend')
      return
    end if
    if (size(input) < 1 .or. size(output) < 1) then
      call set_status(status, fortonnx_invalid_argument, 'tensor arrays must not be empty')
      return
    end if
    call reverse_shape(input, input_shape)
    call reverse_shape(output, output_shape)
    call apply_legacy_vector_shape(self, name, input_shape, output_shape)
    call runtime_bind_pointer_shapes(self, name, host_data_pointer(input), host_data_pointer(output), &
        input_shape, output_shape, status, fortonnx_float64, fortonnx_float32)
  end subroutine runtime_bind_host_float64_float32

  subroutine runtime_bind_host_float64_float64(self, name, input, output, status)
    class(fortonnx_runtime), intent(inout) :: self
    character(len=*), intent(in) :: name
    real(c_double), contiguous, target, intent(inout) :: input(..)
    real(c_double), contiguous, target, intent(inout) :: output(..)
    type(fortonnx_status), intent(inout) :: status
    integer(c_int64_t), allocatable :: input_shape(:), output_shape(:)

    call status%clear()
    if (self%options%backend /= fortonnx_cpu) then
      call set_status(status, fortonnx_invalid_argument, 'host buffers require the CPU backend')
      return
    end if
    if (size(input) < 1 .or. size(output) < 1) then
      call set_status(status, fortonnx_invalid_argument, 'tensor arrays must not be empty')
      return
    end if
    call reverse_shape(input, input_shape)
    call reverse_shape(output, output_shape)
    call apply_legacy_vector_shape(self, name, input_shape, output_shape)
    call runtime_bind_pointer_shapes(self, name, host_data_pointer(input), host_data_pointer(output), &
        input_shape, output_shape, status, fortonnx_float64, fortonnx_float64)
  end subroutine runtime_bind_host_float64_float64

  subroutine runtime_bind_host_float64_bool(self, name, input, output, status)
    class(fortonnx_runtime), intent(inout) :: self
    character(len=*), intent(in) :: name
    real(c_double), contiguous, target, intent(inout) :: input(..)
    logical(c_bool), contiguous, target, intent(inout) :: output(..)
    type(fortonnx_status), intent(inout) :: status
    integer(c_int64_t), allocatable :: input_shape(:), output_shape(:)

    call status%clear()
    if (self%options%backend /= fortonnx_cpu) then
      call set_status(status, fortonnx_invalid_argument, 'host buffers require the CPU backend')
      return
    end if
    if (size(input) < 1 .or. size(output) < 1) then
      call set_status(status, fortonnx_invalid_argument, 'tensor arrays must not be empty')
      return
    end if
    call reverse_shape(input, input_shape)
    call reverse_shape(output, output_shape)
    call apply_legacy_vector_shape(self, name, input_shape, output_shape)
    call runtime_bind_pointer_shapes(self, name, host_data_pointer(input), host_data_pointer(output), &
        input_shape, output_shape, status, fortonnx_float64, fortonnx_bool)
  end subroutine runtime_bind_host_float64_bool

  subroutine runtime_bind_host_bool_float32(self, name, input, output, status)
    class(fortonnx_runtime), intent(inout) :: self
    character(len=*), intent(in) :: name
    logical(c_bool), contiguous, target, intent(inout) :: input(..)
    real(c_float), contiguous, target, intent(inout) :: output(..)
    type(fortonnx_status), intent(inout) :: status
    integer(c_int64_t), allocatable :: input_shape(:), output_shape(:)

    call status%clear()
    if (self%options%backend /= fortonnx_cpu) then
      call set_status(status, fortonnx_invalid_argument, 'host buffers require the CPU backend')
      return
    end if
    if (size(input) < 1 .or. size(output) < 1) then
      call set_status(status, fortonnx_invalid_argument, 'tensor arrays must not be empty')
      return
    end if
    call reverse_shape(input, input_shape)
    call reverse_shape(output, output_shape)
    call apply_legacy_vector_shape(self, name, input_shape, output_shape)
    call runtime_bind_pointer_shapes(self, name, host_data_pointer(input), host_data_pointer(output), &
        input_shape, output_shape, status, fortonnx_bool, fortonnx_float32)
  end subroutine runtime_bind_host_bool_float32

  subroutine runtime_bind_host_bool_float64(self, name, input, output, status)
    class(fortonnx_runtime), intent(inout) :: self
    character(len=*), intent(in) :: name
    logical(c_bool), contiguous, target, intent(inout) :: input(..)
    real(c_double), contiguous, target, intent(inout) :: output(..)
    type(fortonnx_status), intent(inout) :: status
    integer(c_int64_t), allocatable :: input_shape(:), output_shape(:)

    call status%clear()
    if (self%options%backend /= fortonnx_cpu) then
      call set_status(status, fortonnx_invalid_argument, 'host buffers require the CPU backend')
      return
    end if
    if (size(input) < 1 .or. size(output) < 1) then
      call set_status(status, fortonnx_invalid_argument, 'tensor arrays must not be empty')
      return
    end if
    call reverse_shape(input, input_shape)
    call reverse_shape(output, output_shape)
    call apply_legacy_vector_shape(self, name, input_shape, output_shape)
    call runtime_bind_pointer_shapes(self, name, host_data_pointer(input), host_data_pointer(output), &
        input_shape, output_shape, status, fortonnx_bool, fortonnx_float64)
  end subroutine runtime_bind_host_bool_float64

  subroutine runtime_bind_host_bool_bool(self, name, input, output, status)
    class(fortonnx_runtime), intent(inout) :: self
    character(len=*), intent(in) :: name
    logical(c_bool), contiguous, target, intent(inout) :: input(..)
    logical(c_bool), contiguous, target, intent(inout) :: output(..)
    type(fortonnx_status), intent(inout) :: status
    integer(c_int64_t), allocatable :: input_shape(:), output_shape(:)

    call status%clear()
    if (self%options%backend /= fortonnx_cpu) then
      call set_status(status, fortonnx_invalid_argument, 'host buffers require the CPU backend')
      return
    end if
    if (size(input) < 1 .or. size(output) < 1) then
      call set_status(status, fortonnx_invalid_argument, 'tensor arrays must not be empty')
      return
    end if
    call reverse_shape(input, input_shape)
    call reverse_shape(output, output_shape)
    call apply_legacy_vector_shape(self, name, input_shape, output_shape)
    call runtime_bind_pointer_shapes(self, name, host_data_pointer(input), host_data_pointer(output), &
        input_shape, output_shape, status, fortonnx_bool, fortonnx_bool)
  end subroutine runtime_bind_host_bool_bool

  subroutine runtime_bind_pointer_batch(self, name, input, output, batch_size, status, input_type, output_type)
    class(fortonnx_runtime), intent(inout) :: self
    character(len=*), intent(in) :: name
    type(c_ptr), intent(in) :: input, output
    integer(c_int64_t), intent(in) :: batch_size
    type(fortonnx_status), intent(inout) :: status
    integer(c_int), intent(in), optional :: input_type, output_type
    integer :: index
    integer(c_int64_t) :: input_shape(2), output_shape(2)

    call status%clear()
    index = self%find(name)
    if (index == 0) then
      call set_status(status, fortonnx_invalid_argument, 'unknown model: ' // trim(name))
      return
    end if
    if (batch_size < 1) then
      call set_status(status, fortonnx_invalid_argument, 'batch size must be positive')
      return
    end if
    if (size(self%models(index)%inputs) /= 1 .or. &
        size(self%models(index)%outputs) /= 1 .or. &
        size(self%models(index)%inputs(1)%shape) /= 2 .or. &
        size(self%models(index)%outputs(1)%shape) /= 2 .or. &
        self%models(index)%inputs(1)%shape(2) <= 0 .or. &
        self%models(index)%outputs(1)%shape(2) <= 0) then
      call set_status(status, fortonnx_invalid_argument, &
          'batch-only pointer binding requires rank-2 tensors with static feature dimensions')
      return
    end if
    input_shape = [batch_size, self%models(index)%inputs(1)%shape(2)]
    output_shape = [batch_size, self%models(index)%outputs(1)%shape(2)]
    call runtime_bind_pointer_shapes(self, name, input, output, &
        input_shape, output_shape, status, input_type, output_type)
  end subroutine runtime_bind_pointer_batch

  subroutine runtime_bind_pointer_shapes(self, name, input, output, &
      input_shape, output_shape, status, input_type, output_type)
    class(fortonnx_runtime), intent(inout) :: self
    character(len=*), intent(in) :: name
    type(c_ptr), intent(in) :: input, output
    integer(c_int64_t), contiguous, intent(in) :: input_shape(:), output_shape(:)
    type(fortonnx_status), intent(inout) :: status
    integer(c_int), intent(in), optional :: input_type, output_type
    integer(c_int) :: code, in_type, out_type
    integer :: index

    call status%clear()
    if (.not. c_associated(input) .or. .not. c_associated(output)) then
      call set_status(status, fortonnx_invalid_argument, 'tensor buffer pointer is null')
      return
    end if
    if (size(input_shape) < 1 .or. size(output_shape) < 1 .or. &
        any(input_shape <= 0_c_int64_t) .or. any(output_shape <= 0_c_int64_t)) then
      call set_status(status, fortonnx_invalid_argument, &
          'bound tensor shapes must have positive dimensions')
      return
    end if
    index = self%find(name)
    if (index == 0) then
      call set_status(status, fortonnx_invalid_argument, 'unknown model: ' // trim(name))
      return
    end if
    self%models(index)%bound = .false.
    in_type = fortonnx_float32
    out_type = fortonnx_float32
    if (present(input_type)) in_type = input_type
    if (present(output_type)) out_type = output_type
    code = c_fortonnx_session_bind_tensor_typed(self%models(index)%session, input, output, &
        self%options%backend, input_shape, int(size(input_shape), c_int64_t), &
        output_shape, int(size(output_shape), c_int64_t), in_type, out_type)
    if (code /= 0_c_int) then
      call set_c_error(status, 'failed to bind tensor buffers', self%models(index)%session)
      return
    end if
    self%models(index)%batch_size = input_shape(1)
    self%models(index)%bound = .true.
  end subroutine runtime_bind_pointer_shapes

  subroutine runtime_bind_tensor_pair(self, name, input, output, status)
    class(fortonnx_runtime), intent(inout) :: self
    character(len=*), intent(in) :: name
    type(fortonnx_tensor), intent(in) :: input, output
    type(fortonnx_status), intent(inout) :: status
    type(fortonnx_tensor) :: input_view, output_view

    input_view = input
    output_view = output
    if (allocated(input_view%shape) .and. allocated(output_view%shape)) then
      call apply_legacy_vector_shape(self, name, input_view%shape, output_view%shape)
    end if
    call runtime_bind_tensor_arrays(self, name, [input_view], [output_view], status)
  end subroutine runtime_bind_tensor_pair

  subroutine runtime_bind_tensor_inputs(self, name, inputs, output, status)
    class(fortonnx_runtime), intent(inout) :: self
    character(len=*), intent(in) :: name
    type(fortonnx_tensor), intent(in) :: inputs(:)
    type(fortonnx_tensor), intent(in) :: output
    type(fortonnx_status), intent(inout) :: status

    call runtime_bind_tensor_arrays(self, name, inputs, [output], status)
  end subroutine runtime_bind_tensor_inputs

  logical function tensor_memory_matches(runtime_backend, tensor_backend)
    integer(c_int), intent(in) :: runtime_backend, tensor_backend
    if (tensor_backend < fortonnx_cpu .or. tensor_backend > fortonnx_tensorrt) then
      tensor_memory_matches = .false.
      return
    end if
    tensor_memory_matches = (runtime_backend == fortonnx_cpu .and. &
        tensor_backend == fortonnx_cpu) .or. &
        (runtime_backend /= fortonnx_cpu .and. tensor_backend /= fortonnx_cpu)
  end function tensor_memory_matches

  subroutine resolve_tensor_index(session, is_input, tensor, position, &
      resolved, status)
    type(c_ptr), intent(in) :: session
    integer(c_int), intent(in) :: is_input
    type(fortonnx_tensor), intent(in) :: tensor
    integer, intent(in) :: position
    integer(c_int64_t), intent(out) :: resolved
    type(fortonnx_status), intent(inout) :: status
    character(kind=c_char, len=:), allocatable :: c_name
    integer(c_int) :: code

    if (allocated(tensor%name)) then
      if (len_trim(tensor%name) > 0) then
        c_name = trim(tensor%name) // c_null_char
        code = c_fortonnx_session_find_tensor(session, is_input, c_name, resolved)
        if (code /= 0_c_int) then
          call set_c_error(status, 'failed to resolve tensor ' // trim(tensor%name), session)
        end if
        return
      end if
    end if
    resolved = int(position - 1, c_int64_t)
  end subroutine resolve_tensor_index

  subroutine validate_tensor_view(self, tensor, status)
    class(fortonnx_runtime), intent(in) :: self
    type(fortonnx_tensor), intent(in) :: tensor
    type(fortonnx_status), intent(inout) :: status

    if (.not. c_associated(tensor%data)) then
      call set_status(status, fortonnx_invalid_argument, 'tensor view data pointer is null')
    else if (.not. allocated(tensor%shape)) then
      call set_status(status, fortonnx_invalid_argument, 'tensor view has no shape')
    else if (size(tensor%shape) < 1 .or. any(tensor%shape <= 0_c_int64_t)) then
      call set_status(status, fortonnx_invalid_argument, &
          'tensor view shape must have positive dimensions')
    else if (.not. tensor_memory_matches(self%options%backend, tensor%memory_backend)) then
      call set_status(status, fortonnx_invalid_argument, &
          'tensor view memory does not match the inference backend')
    end if
  end subroutine validate_tensor_view

  subroutine runtime_bind_tensor_arrays(self, name, inputs, outputs, status)
    class(fortonnx_runtime), intent(inout) :: self
    character(len=*), intent(in) :: name
    type(fortonnx_tensor), intent(in) :: inputs(:), outputs(:)
    type(fortonnx_status), intent(inout) :: status
    logical, allocatable :: input_seen(:), output_seen(:)
    integer(c_int64_t) :: tensor_index
    integer(c_int) :: code
    integer :: index, position

    call status%clear()
    index = self%find(name)
    if (index == 0) then
      call set_status(status, fortonnx_invalid_argument, 'unknown model: ' // trim(name))
      return
    end if
    self%models(index)%bound = .false.
    if (size(inputs) /= size(self%models(index)%inputs) .or. &
        size(outputs) /= size(self%models(index)%outputs)) then
      call set_status(status, fortonnx_invalid_argument, &
          'tensor view counts do not match the ONNX model inputs and outputs')
      return
    end if
    do position = 1, size(inputs)
      call validate_tensor_view(self, inputs(position), status)
      if (.not. status%ok()) return
    end do
    do position = 1, size(outputs)
      call validate_tensor_view(self, outputs(position), status)
      if (.not. status%ok()) return
    end do

    code = c_fortonnx_session_begin_bind(self%models(index)%session)
    if (code /= 0_c_int) then
      call set_c_error(status, 'failed to begin tensor binding', self%models(index)%session)
      return
    end if
    allocate(input_seen(size(inputs)), output_seen(size(outputs)))
    input_seen = .false.
    output_seen = .false.
    do position = 1, size(inputs)
      call resolve_tensor_index(self%models(index)%session, fortonnx_input, &
          inputs(position), position, tensor_index, status)
      if (.not. status%ok()) return
      if (tensor_index < 0_c_int64_t .or. tensor_index >= int(size(inputs), c_int64_t)) then
        call set_status(status, fortonnx_invalid_argument, 'input tensor index is out of range')
        return
      end if
      if (input_seen(int(tensor_index) + 1)) then
        call set_status(status, fortonnx_invalid_argument, 'an input tensor was specified twice')
        return
      end if
      input_seen(int(tensor_index) + 1) = .true.
      code = c_fortonnx_session_bind_input_typed(self%models(index)%session, tensor_index, &
          inputs(position)%data, inputs(position)%memory_backend, inputs(position)%shape, &
          int(size(inputs(position)%shape), c_int64_t), inputs(position)%element_type)
      if (code /= 0_c_int) then
        call set_c_error(status, 'failed to bind an input tensor', self%models(index)%session)
        return
      end if
    end do
    do position = 1, size(outputs)
      call resolve_tensor_index(self%models(index)%session, fortonnx_output, &
          outputs(position), position, tensor_index, status)
      if (.not. status%ok()) return
      if (tensor_index < 0_c_int64_t .or. tensor_index >= int(size(outputs), c_int64_t)) then
        call set_status(status, fortonnx_invalid_argument, 'output tensor index is out of range')
        return
      end if
      if (output_seen(int(tensor_index) + 1)) then
        call set_status(status, fortonnx_invalid_argument, 'an output tensor was specified twice')
        return
      end if
      output_seen(int(tensor_index) + 1) = .true.
      code = c_fortonnx_session_bind_output_typed(self%models(index)%session, tensor_index, &
          outputs(position)%data, outputs(position)%memory_backend, outputs(position)%shape, &
          int(size(outputs(position)%shape), c_int64_t), outputs(position)%element_type)
      if (code /= 0_c_int) then
        call set_c_error(status, 'failed to bind an output tensor', self%models(index)%session)
        return
      end if
    end do
    self%models(index)%batch_size = inputs(1)%shape(1)
    self%models(index)%bound = .true.
  end subroutine runtime_bind_tensor_arrays

#ifdef FORTONNX_WITH_CUDA_FORTRAN
  subroutine set_cuda_status(status, code, operation)
    type(fortonnx_status), intent(inout) :: status
    integer, intent(in) :: code
    character(len=*), intent(in) :: operation
    character(len=32) :: code_text

    if (code == 0) then
      call status%clear()
    else
      write(code_text, '(i0)') code
      status%code = fortonnx_runtime_error
      status%message = trim(operation) // ' failed with CUDA status ' // trim(code_text)
    end if
  end subroutine set_cuda_status

  subroutine stream_create(self, status)
    class(fortonnx_cuda_stream), intent(inout) :: self
    type(fortonnx_status), intent(inout) :: status
    integer :: code

    if (self%active) call self%close(status)
    code = cudaStreamCreate(self%handle)
    call set_cuda_status(status, code, 'cudaStreamCreate')
    self%active = status%ok()
  end subroutine stream_create

  subroutine stream_synchronize(self, status)
    class(fortonnx_cuda_stream), intent(inout) :: self
    type(fortonnx_status), intent(inout) :: status
    integer :: code

    if (.not. self%active) then
      status%code = fortonnx_invalid_argument
      status%message = 'CUDA stream is not active'
      return
    end if
    code = cudaStreamSynchronize(self%handle)
    call set_cuda_status(status, code, 'cudaStreamSynchronize')
  end subroutine stream_synchronize

  function stream_c_pointer(self) result(pointer)
    class(fortonnx_cuda_stream), intent(in) :: self
    type(c_ptr) :: pointer

    if (self%active) then
      pointer = transfer(self%handle, c_null_ptr)
    else
      pointer = c_null_ptr
    end if
  end function stream_c_pointer

  subroutine stream_close(self, status)
    class(fortonnx_cuda_stream), intent(inout) :: self
    type(fortonnx_status), intent(inout) :: status
    integer :: code

    call status%clear()
    if (.not. self%active) return
    code = cudaStreamSynchronize(self%handle)
    if (code == 0) code = cudaStreamDestroy(self%handle)
    call set_cuda_status(status, code, 'CUDA stream shutdown')
    self%handle = 0
    self%active = .false.
  end subroutine stream_close

  subroutine stream_finalize(self)
    type(fortonnx_cuda_stream), intent(inout) :: self
    integer :: ignored

    if (.not. self%active) return
    ignored = cudaStreamSynchronize(self%handle)
    ignored = cudaStreamDestroy(self%handle)
    self%handle = 0
    self%active = .false.
  end subroutine stream_finalize

  function make_device_tensor_1d(array, name) result(tensor)
    real(c_float), device, target, contiguous, intent(inout) :: array(:)
    character(len=*), intent(in), optional :: name
    type(fortonnx_tensor) :: tensor
    tensor%data = transfer(c_devloc(array), c_null_ptr)
    tensor%shape = [int(size(array, 1), c_int64_t)]
    tensor%memory_backend = fortonnx_cuda
    if (present(name)) tensor%name = trim(name)
  end function make_device_tensor_1d

  function make_device_tensor_1d_float64(array, name) result(tensor)
    real(c_double), device, target, contiguous, intent(inout) :: array(:)
    character(len=*), intent(in), optional :: name
    type(fortonnx_tensor) :: tensor
    tensor%element_type = fortonnx_float64
    tensor%data = transfer(c_devloc(array), c_null_ptr)
    tensor%shape = [int(size(array, 1), c_int64_t)]
    tensor%memory_backend = fortonnx_cuda
    if (present(name)) tensor%name = trim(name)
  end function make_device_tensor_1d_float64

  function make_device_tensor_1d_bool(array, name) result(tensor)
    logical(c_bool), device, target, contiguous, intent(inout) :: array(:)
    character(len=*), intent(in), optional :: name
    type(fortonnx_tensor) :: tensor
    tensor%element_type = fortonnx_bool
    tensor%data = transfer(c_devloc(array), c_null_ptr)
    tensor%shape = [int(size(array, 1), c_int64_t)]
    tensor%memory_backend = fortonnx_cuda
    if (present(name)) tensor%name = trim(name)
  end function make_device_tensor_1d_bool

  function make_device_tensor_2d(array, name) result(tensor)
    real(c_float), device, target, contiguous, intent(inout) :: array(:, :)
    character(len=*), intent(in), optional :: name
    type(fortonnx_tensor) :: tensor
    tensor%data = transfer(c_devloc(array), c_null_ptr)
    tensor%shape = [int(size(array, 2), c_int64_t), int(size(array, 1), c_int64_t)]
    tensor%memory_backend = fortonnx_cuda
    if (present(name)) tensor%name = trim(name)
  end function make_device_tensor_2d

  function make_device_tensor_2d_float64(array, name) result(tensor)
    real(c_double), device, target, contiguous, intent(inout) :: array(:, :)
    character(len=*), intent(in), optional :: name
    type(fortonnx_tensor) :: tensor
    tensor%element_type = fortonnx_float64
    tensor%data = transfer(c_devloc(array), c_null_ptr)
    tensor%shape = [int(size(array, 2), c_int64_t), int(size(array, 1), c_int64_t)]
    tensor%memory_backend = fortonnx_cuda
    if (present(name)) tensor%name = trim(name)
  end function make_device_tensor_2d_float64

  function make_device_tensor_2d_bool(array, name) result(tensor)
    logical(c_bool), device, target, contiguous, intent(inout) :: array(:, :)
    character(len=*), intent(in), optional :: name
    type(fortonnx_tensor) :: tensor
    tensor%element_type = fortonnx_bool
    tensor%data = transfer(c_devloc(array), c_null_ptr)
    tensor%shape = [int(size(array, 2), c_int64_t), int(size(array, 1), c_int64_t)]
    tensor%memory_backend = fortonnx_cuda
    if (present(name)) tensor%name = trim(name)
  end function make_device_tensor_2d_bool

  function make_device_tensor_3d(array, name) result(tensor)
    real(c_float), device, target, contiguous, intent(inout) :: array(:, :, :)
    character(len=*), intent(in), optional :: name
    type(fortonnx_tensor) :: tensor
    tensor%data = transfer(c_devloc(array), c_null_ptr)
    tensor%shape = [int(size(array, 3), c_int64_t), int(size(array, 2), c_int64_t), &
        int(size(array, 1), c_int64_t)]
    tensor%memory_backend = fortonnx_cuda
    if (present(name)) tensor%name = trim(name)
  end function make_device_tensor_3d

  function make_device_tensor_3d_float64(array, name) result(tensor)
    real(c_double), device, target, contiguous, intent(inout) :: array(:, :, :)
    character(len=*), intent(in), optional :: name
    type(fortonnx_tensor) :: tensor
    tensor%element_type = fortonnx_float64
    tensor%data = transfer(c_devloc(array), c_null_ptr)
    tensor%shape = [int(size(array, 3), c_int64_t), int(size(array, 2), c_int64_t), &
        int(size(array, 1), c_int64_t)]
    tensor%memory_backend = fortonnx_cuda
    if (present(name)) tensor%name = trim(name)
  end function make_device_tensor_3d_float64

  function make_device_tensor_3d_bool(array, name) result(tensor)
    logical(c_bool), device, target, contiguous, intent(inout) :: array(:, :, :)
    character(len=*), intent(in), optional :: name
    type(fortonnx_tensor) :: tensor
    tensor%element_type = fortonnx_bool
    tensor%data = transfer(c_devloc(array), c_null_ptr)
    tensor%shape = [int(size(array, 3), c_int64_t), int(size(array, 2), c_int64_t), &
        int(size(array, 1), c_int64_t)]
    tensor%memory_backend = fortonnx_cuda
    if (present(name)) tensor%name = trim(name)
  end function make_device_tensor_3d_bool

  function make_device_tensor_4d(array, name) result(tensor)
    real(c_float), device, target, contiguous, intent(inout) :: array(:, :, :, :)
    character(len=*), intent(in), optional :: name
    type(fortonnx_tensor) :: tensor
    tensor%data = transfer(c_devloc(array), c_null_ptr)
    tensor%shape = [int(size(array, 4), c_int64_t), int(size(array, 3), c_int64_t), &
        int(size(array, 2), c_int64_t), int(size(array, 1), c_int64_t)]
    tensor%memory_backend = fortonnx_cuda
    if (present(name)) tensor%name = trim(name)
  end function make_device_tensor_4d

  function make_device_tensor_4d_float64(array, name) result(tensor)
    real(c_double), device, target, contiguous, intent(inout) :: array(:, :, :, :)
    character(len=*), intent(in), optional :: name
    type(fortonnx_tensor) :: tensor
    tensor%element_type = fortonnx_float64
    tensor%data = transfer(c_devloc(array), c_null_ptr)
    tensor%shape = [int(size(array, 4), c_int64_t), int(size(array, 3), c_int64_t), &
        int(size(array, 2), c_int64_t), int(size(array, 1), c_int64_t)]
    tensor%memory_backend = fortonnx_cuda
    if (present(name)) tensor%name = trim(name)
  end function make_device_tensor_4d_float64

  function make_device_tensor_4d_bool(array, name) result(tensor)
    logical(c_bool), device, target, contiguous, intent(inout) :: array(:, :, :, :)
    character(len=*), intent(in), optional :: name
    type(fortonnx_tensor) :: tensor
    tensor%element_type = fortonnx_bool
    tensor%data = transfer(c_devloc(array), c_null_ptr)
    tensor%shape = [int(size(array, 4), c_int64_t), int(size(array, 3), c_int64_t), &
        int(size(array, 2), c_int64_t), int(size(array, 1), c_int64_t)]
    tensor%memory_backend = fortonnx_cuda
    if (present(name)) tensor%name = trim(name)
  end function make_device_tensor_4d_bool

  function make_device_tensor_5d(array, name) result(tensor)
    real(c_float), device, target, contiguous, intent(inout) :: array(:, :, :, :, :)
    character(len=*), intent(in), optional :: name
    type(fortonnx_tensor) :: tensor
    tensor%data = transfer(c_devloc(array), c_null_ptr)
    tensor%shape = [int(size(array, 5), c_int64_t), int(size(array, 4), c_int64_t), &
        int(size(array, 3), c_int64_t), int(size(array, 2), c_int64_t), &
        int(size(array, 1), c_int64_t)]
    tensor%memory_backend = fortonnx_cuda
    if (present(name)) tensor%name = trim(name)
  end function make_device_tensor_5d

  function make_device_tensor_5d_float64(array, name) result(tensor)
    real(c_double), device, target, contiguous, intent(inout) :: array(:, :, :, :, :)
    character(len=*), intent(in), optional :: name
    type(fortonnx_tensor) :: tensor
    tensor%element_type = fortonnx_float64
    tensor%data = transfer(c_devloc(array), c_null_ptr)
    tensor%shape = [int(size(array, 5), c_int64_t), int(size(array, 4), c_int64_t), &
        int(size(array, 3), c_int64_t), int(size(array, 2), c_int64_t), &
        int(size(array, 1), c_int64_t)]
    tensor%memory_backend = fortonnx_cuda
    if (present(name)) tensor%name = trim(name)
  end function make_device_tensor_5d_float64

  function make_device_tensor_5d_bool(array, name) result(tensor)
    logical(c_bool), device, target, contiguous, intent(inout) :: array(:, :, :, :, :)
    character(len=*), intent(in), optional :: name
    type(fortonnx_tensor) :: tensor
    tensor%element_type = fortonnx_bool
    tensor%data = transfer(c_devloc(array), c_null_ptr)
    tensor%shape = [int(size(array, 5), c_int64_t), int(size(array, 4), c_int64_t), &
        int(size(array, 3), c_int64_t), int(size(array, 2), c_int64_t), &
        int(size(array, 1), c_int64_t)]
    tensor%memory_backend = fortonnx_cuda
    if (present(name)) tensor%name = trim(name)
  end function make_device_tensor_5d_bool

  function make_device_tensor_6d(array, name) result(tensor)
    real(c_float), device, target, contiguous, intent(inout) :: array(:, :, :, :, :, :)
    character(len=*), intent(in), optional :: name
    type(fortonnx_tensor) :: tensor
    tensor%data = transfer(c_devloc(array), c_null_ptr)
    tensor%shape = [int(size(array, 6), c_int64_t), int(size(array, 5), c_int64_t), &
        int(size(array, 4), c_int64_t), int(size(array, 3), c_int64_t), &
        int(size(array, 2), c_int64_t), int(size(array, 1), c_int64_t)]
    tensor%memory_backend = fortonnx_cuda
    if (present(name)) tensor%name = trim(name)
  end function make_device_tensor_6d

  function make_device_tensor_6d_float64(array, name) result(tensor)
    real(c_double), device, target, contiguous, intent(inout) :: array(:, :, :, :, :, :)
    character(len=*), intent(in), optional :: name
    type(fortonnx_tensor) :: tensor
    tensor%element_type = fortonnx_float64
    tensor%data = transfer(c_devloc(array), c_null_ptr)
    tensor%shape = [int(size(array, 6), c_int64_t), int(size(array, 5), c_int64_t), &
        int(size(array, 4), c_int64_t), int(size(array, 3), c_int64_t), &
        int(size(array, 2), c_int64_t), int(size(array, 1), c_int64_t)]
    tensor%memory_backend = fortonnx_cuda
    if (present(name)) tensor%name = trim(name)
  end function make_device_tensor_6d_float64

  function make_device_tensor_6d_bool(array, name) result(tensor)
    logical(c_bool), device, target, contiguous, intent(inout) :: array(:, :, :, :, :, :)
    character(len=*), intent(in), optional :: name
    type(fortonnx_tensor) :: tensor
    tensor%element_type = fortonnx_bool
    tensor%data = transfer(c_devloc(array), c_null_ptr)
    tensor%shape = [int(size(array, 6), c_int64_t), int(size(array, 5), c_int64_t), &
        int(size(array, 4), c_int64_t), int(size(array, 3), c_int64_t), &
        int(size(array, 2), c_int64_t), int(size(array, 1), c_int64_t)]
    tensor%memory_backend = fortonnx_cuda
    if (present(name)) tensor%name = trim(name)
  end function make_device_tensor_6d_bool

  function make_device_tensor_7d(array, name) result(tensor)
    real(c_float), device, target, contiguous, intent(inout) :: array(:, :, :, :, :, :, :)
    character(len=*), intent(in), optional :: name
    type(fortonnx_tensor) :: tensor
    tensor%data = transfer(c_devloc(array), c_null_ptr)
    tensor%shape = [int(size(array, 7), c_int64_t), int(size(array, 6), c_int64_t), &
        int(size(array, 5), c_int64_t), int(size(array, 4), c_int64_t), &
        int(size(array, 3), c_int64_t), int(size(array, 2), c_int64_t), &
        int(size(array, 1), c_int64_t)]
    tensor%memory_backend = fortonnx_cuda
    if (present(name)) tensor%name = trim(name)
  end function make_device_tensor_7d

  function make_device_tensor_7d_float64(array, name) result(tensor)
    real(c_double), device, target, contiguous, intent(inout) :: array(:, :, :, :, :, :, :)
    character(len=*), intent(in), optional :: name
    type(fortonnx_tensor) :: tensor
    tensor%element_type = fortonnx_float64
    tensor%data = transfer(c_devloc(array), c_null_ptr)
    tensor%shape = [int(size(array, 7), c_int64_t), int(size(array, 6), c_int64_t), &
        int(size(array, 5), c_int64_t), int(size(array, 4), c_int64_t), &
        int(size(array, 3), c_int64_t), int(size(array, 2), c_int64_t), &
        int(size(array, 1), c_int64_t)]
    tensor%memory_backend = fortonnx_cuda
    if (present(name)) tensor%name = trim(name)
  end function make_device_tensor_7d_float64

  function make_device_tensor_7d_bool(array, name) result(tensor)
    logical(c_bool), device, target, contiguous, intent(inout) :: array(:, :, :, :, :, :, :)
    character(len=*), intent(in), optional :: name
    type(fortonnx_tensor) :: tensor
    tensor%element_type = fortonnx_bool
    tensor%data = transfer(c_devloc(array), c_null_ptr)
    tensor%shape = [int(size(array, 7), c_int64_t), int(size(array, 6), c_int64_t), &
        int(size(array, 5), c_int64_t), int(size(array, 4), c_int64_t), &
        int(size(array, 3), c_int64_t), int(size(array, 2), c_int64_t), &
        int(size(array, 1), c_int64_t)]
    tensor%memory_backend = fortonnx_cuda
    if (present(name)) tensor%name = trim(name)
  end function make_device_tensor_7d_bool

  subroutine runtime_bind_device_array_1d(self, name, input, output, status)
    class(fortonnx_runtime), intent(inout) :: self
    character(len=*), intent(in) :: name
    real(c_float), device, target, contiguous, intent(inout) :: input(:), output(:)
    type(fortonnx_status), intent(inout) :: status
    call runtime_bind_tensor_pair(self, name, fortonnx_device_tensor(input), &
        fortonnx_device_tensor(output), status)
  end subroutine runtime_bind_device_array_1d

  subroutine runtime_bind_device_array_2d(self, name, input, output, status)
    class(fortonnx_runtime), intent(inout) :: self
    character(len=*), intent(in) :: name
    real(c_float), device, target, contiguous, intent(inout) :: input(:, :), output(:, :)
    type(fortonnx_status), intent(inout) :: status
    call runtime_bind_tensor_pair(self, name, fortonnx_device_tensor(input), &
        fortonnx_device_tensor(output), status)
  end subroutine runtime_bind_device_array_2d

  subroutine runtime_bind_device_array_3d(self, name, input, output, status)
    class(fortonnx_runtime), intent(inout) :: self
    character(len=*), intent(in) :: name
    real(c_float), device, target, contiguous, intent(inout) :: input(:, :, :), output(:, :, :)
    type(fortonnx_status), intent(inout) :: status
    call runtime_bind_tensor_pair(self, name, fortonnx_device_tensor(input), &
        fortonnx_device_tensor(output), status)
  end subroutine runtime_bind_device_array_3d

  subroutine runtime_bind_device_array_4d(self, name, input, output, status)
    class(fortonnx_runtime), intent(inout) :: self
    character(len=*), intent(in) :: name
    real(c_float), device, target, contiguous, intent(inout) :: input(:, :, :, :), output(:, :, :, :)
    type(fortonnx_status), intent(inout) :: status
    call runtime_bind_tensor_pair(self, name, fortonnx_device_tensor(input), &
        fortonnx_device_tensor(output), status)
  end subroutine runtime_bind_device_array_4d

  subroutine runtime_bind_device_array_5d(self, name, input, output, status)
    class(fortonnx_runtime), intent(inout) :: self
    character(len=*), intent(in) :: name
    real(c_float), device, target, contiguous, intent(inout) :: input(:, :, :, :, :), output(:, :, :, :, :)
    type(fortonnx_status), intent(inout) :: status
    call runtime_bind_tensor_pair(self, name, fortonnx_device_tensor(input), &
        fortonnx_device_tensor(output), status)
  end subroutine runtime_bind_device_array_5d

  subroutine runtime_bind_device_array_6d(self, name, input, output, status)
    class(fortonnx_runtime), intent(inout) :: self
    character(len=*), intent(in) :: name
    real(c_float), device, target, contiguous, intent(inout) :: input(:, :, :, :, :, :), output(:, :, :, :, :, :)
    type(fortonnx_status), intent(inout) :: status
    call runtime_bind_tensor_pair(self, name, fortonnx_device_tensor(input), &
        fortonnx_device_tensor(output), status)
  end subroutine runtime_bind_device_array_6d

  subroutine runtime_bind_device_array_7d(self, name, input, output, status)
    class(fortonnx_runtime), intent(inout) :: self
    character(len=*), intent(in) :: name
    real(c_float), device, target, contiguous, intent(inout) :: input(:, :, :, :, :, :, :)
    real(c_float), device, target, contiguous, intent(inout) :: output(:, :, :, :, :, :, :)
    type(fortonnx_status), intent(inout) :: status
    call runtime_bind_tensor_pair(self, name, fortonnx_device_tensor(input), &
        fortonnx_device_tensor(output), status)
  end subroutine runtime_bind_device_array_7d

  subroutine runtime_bind_device_7d_bool_bool(self, name, input, output, status)
    class(fortonnx_runtime), intent(inout) :: self
    character(len=*), intent(in) :: name
    logical(c_bool), device, target, contiguous, intent(inout) :: input(:, :, :, :, :, :, :)
    logical(c_bool), device, target, contiguous, intent(inout) :: output(:, :, :, :, :, :, :)
    type(fortonnx_status), intent(inout) :: status
    call runtime_bind_tensor_pair(self, name, fortonnx_device_tensor(input), &
        fortonnx_device_tensor(output), status)
  end subroutine runtime_bind_device_7d_bool_bool

  subroutine runtime_bind_device_7d_bool_float64(self, name, input, output, status)
    class(fortonnx_runtime), intent(inout) :: self
    character(len=*), intent(in) :: name
    logical(c_bool), device, target, contiguous, intent(inout) :: input(:, :, :, :, :, :, :)
    real(c_double), device, target, contiguous, intent(inout) :: output(:, :, :, :, :, :, :)
    type(fortonnx_status), intent(inout) :: status
    call runtime_bind_tensor_pair(self, name, fortonnx_device_tensor(input), &
        fortonnx_device_tensor(output), status)
  end subroutine runtime_bind_device_7d_bool_float64

  subroutine runtime_bind_device_7d_bool_float32(self, name, input, output, status)
    class(fortonnx_runtime), intent(inout) :: self
    character(len=*), intent(in) :: name
    logical(c_bool), device, target, contiguous, intent(inout) :: input(:, :, :, :, :, :, :)
    real(c_float), device, target, contiguous, intent(inout) :: output(:, :, :, :, :, :, :)
    type(fortonnx_status), intent(inout) :: status
    call runtime_bind_tensor_pair(self, name, fortonnx_device_tensor(input), &
        fortonnx_device_tensor(output), status)
  end subroutine runtime_bind_device_7d_bool_float32

  subroutine runtime_bind_device_7d_float64_bool(self, name, input, output, status)
    class(fortonnx_runtime), intent(inout) :: self
    character(len=*), intent(in) :: name
    real(c_double), device, target, contiguous, intent(inout) :: input(:, :, :, :, :, :, :)
    logical(c_bool), device, target, contiguous, intent(inout) :: output(:, :, :, :, :, :, :)
    type(fortonnx_status), intent(inout) :: status
    call runtime_bind_tensor_pair(self, name, fortonnx_device_tensor(input), &
        fortonnx_device_tensor(output), status)
  end subroutine runtime_bind_device_7d_float64_bool

  subroutine runtime_bind_device_7d_float64_float64(self, name, input, output, status)
    class(fortonnx_runtime), intent(inout) :: self
    character(len=*), intent(in) :: name
    real(c_double), device, target, contiguous, intent(inout) :: input(:, :, :, :, :, :, :)
    real(c_double), device, target, contiguous, intent(inout) :: output(:, :, :, :, :, :, :)
    type(fortonnx_status), intent(inout) :: status
    call runtime_bind_tensor_pair(self, name, fortonnx_device_tensor(input), &
        fortonnx_device_tensor(output), status)
  end subroutine runtime_bind_device_7d_float64_float64

  subroutine runtime_bind_device_7d_float64_float32(self, name, input, output, status)
    class(fortonnx_runtime), intent(inout) :: self
    character(len=*), intent(in) :: name
    real(c_double), device, target, contiguous, intent(inout) :: input(:, :, :, :, :, :, :)
    real(c_float), device, target, contiguous, intent(inout) :: output(:, :, :, :, :, :, :)
    type(fortonnx_status), intent(inout) :: status
    call runtime_bind_tensor_pair(self, name, fortonnx_device_tensor(input), &
        fortonnx_device_tensor(output), status)
  end subroutine runtime_bind_device_7d_float64_float32

  subroutine runtime_bind_device_7d_float32_bool(self, name, input, output, status)
    class(fortonnx_runtime), intent(inout) :: self
    character(len=*), intent(in) :: name
    real(c_float), device, target, contiguous, intent(inout) :: input(:, :, :, :, :, :, :)
    logical(c_bool), device, target, contiguous, intent(inout) :: output(:, :, :, :, :, :, :)
    type(fortonnx_status), intent(inout) :: status
    call runtime_bind_tensor_pair(self, name, fortonnx_device_tensor(input), &
        fortonnx_device_tensor(output), status)
  end subroutine runtime_bind_device_7d_float32_bool

  subroutine runtime_bind_device_7d_float32_float64(self, name, input, output, status)
    class(fortonnx_runtime), intent(inout) :: self
    character(len=*), intent(in) :: name
    real(c_float), device, target, contiguous, intent(inout) :: input(:, :, :, :, :, :, :)
    real(c_double), device, target, contiguous, intent(inout) :: output(:, :, :, :, :, :, :)
    type(fortonnx_status), intent(inout) :: status
    call runtime_bind_tensor_pair(self, name, fortonnx_device_tensor(input), &
        fortonnx_device_tensor(output), status)
  end subroutine runtime_bind_device_7d_float32_float64

  subroutine runtime_bind_device_6d_bool_bool(self, name, input, output, status)
    class(fortonnx_runtime), intent(inout) :: self
    character(len=*), intent(in) :: name
    logical(c_bool), device, target, contiguous, intent(inout) :: input(:, :, :, :, :, :)
    logical(c_bool), device, target, contiguous, intent(inout) :: output(:, :, :, :, :, :)
    type(fortonnx_status), intent(inout) :: status
    call runtime_bind_tensor_pair(self, name, fortonnx_device_tensor(input), &
        fortonnx_device_tensor(output), status)
  end subroutine runtime_bind_device_6d_bool_bool

  subroutine runtime_bind_device_6d_bool_float64(self, name, input, output, status)
    class(fortonnx_runtime), intent(inout) :: self
    character(len=*), intent(in) :: name
    logical(c_bool), device, target, contiguous, intent(inout) :: input(:, :, :, :, :, :)
    real(c_double), device, target, contiguous, intent(inout) :: output(:, :, :, :, :, :)
    type(fortonnx_status), intent(inout) :: status
    call runtime_bind_tensor_pair(self, name, fortonnx_device_tensor(input), &
        fortonnx_device_tensor(output), status)
  end subroutine runtime_bind_device_6d_bool_float64

  subroutine runtime_bind_device_6d_bool_float32(self, name, input, output, status)
    class(fortonnx_runtime), intent(inout) :: self
    character(len=*), intent(in) :: name
    logical(c_bool), device, target, contiguous, intent(inout) :: input(:, :, :, :, :, :)
    real(c_float), device, target, contiguous, intent(inout) :: output(:, :, :, :, :, :)
    type(fortonnx_status), intent(inout) :: status
    call runtime_bind_tensor_pair(self, name, fortonnx_device_tensor(input), &
        fortonnx_device_tensor(output), status)
  end subroutine runtime_bind_device_6d_bool_float32

  subroutine runtime_bind_device_6d_float64_bool(self, name, input, output, status)
    class(fortonnx_runtime), intent(inout) :: self
    character(len=*), intent(in) :: name
    real(c_double), device, target, contiguous, intent(inout) :: input(:, :, :, :, :, :)
    logical(c_bool), device, target, contiguous, intent(inout) :: output(:, :, :, :, :, :)
    type(fortonnx_status), intent(inout) :: status
    call runtime_bind_tensor_pair(self, name, fortonnx_device_tensor(input), &
        fortonnx_device_tensor(output), status)
  end subroutine runtime_bind_device_6d_float64_bool

  subroutine runtime_bind_device_6d_float64_float64(self, name, input, output, status)
    class(fortonnx_runtime), intent(inout) :: self
    character(len=*), intent(in) :: name
    real(c_double), device, target, contiguous, intent(inout) :: input(:, :, :, :, :, :)
    real(c_double), device, target, contiguous, intent(inout) :: output(:, :, :, :, :, :)
    type(fortonnx_status), intent(inout) :: status
    call runtime_bind_tensor_pair(self, name, fortonnx_device_tensor(input), &
        fortonnx_device_tensor(output), status)
  end subroutine runtime_bind_device_6d_float64_float64

  subroutine runtime_bind_device_6d_float64_float32(self, name, input, output, status)
    class(fortonnx_runtime), intent(inout) :: self
    character(len=*), intent(in) :: name
    real(c_double), device, target, contiguous, intent(inout) :: input(:, :, :, :, :, :)
    real(c_float), device, target, contiguous, intent(inout) :: output(:, :, :, :, :, :)
    type(fortonnx_status), intent(inout) :: status
    call runtime_bind_tensor_pair(self, name, fortonnx_device_tensor(input), &
        fortonnx_device_tensor(output), status)
  end subroutine runtime_bind_device_6d_float64_float32

  subroutine runtime_bind_device_6d_float32_bool(self, name, input, output, status)
    class(fortonnx_runtime), intent(inout) :: self
    character(len=*), intent(in) :: name
    real(c_float), device, target, contiguous, intent(inout) :: input(:, :, :, :, :, :)
    logical(c_bool), device, target, contiguous, intent(inout) :: output(:, :, :, :, :, :)
    type(fortonnx_status), intent(inout) :: status
    call runtime_bind_tensor_pair(self, name, fortonnx_device_tensor(input), &
        fortonnx_device_tensor(output), status)
  end subroutine runtime_bind_device_6d_float32_bool

  subroutine runtime_bind_device_6d_float32_float64(self, name, input, output, status)
    class(fortonnx_runtime), intent(inout) :: self
    character(len=*), intent(in) :: name
    real(c_float), device, target, contiguous, intent(inout) :: input(:, :, :, :, :, :)
    real(c_double), device, target, contiguous, intent(inout) :: output(:, :, :, :, :, :)
    type(fortonnx_status), intent(inout) :: status
    call runtime_bind_tensor_pair(self, name, fortonnx_device_tensor(input), &
        fortonnx_device_tensor(output), status)
  end subroutine runtime_bind_device_6d_float32_float64

  subroutine runtime_bind_device_5d_bool_bool(self, name, input, output, status)
    class(fortonnx_runtime), intent(inout) :: self
    character(len=*), intent(in) :: name
    logical(c_bool), device, target, contiguous, intent(inout) :: input(:, :, :, :, :)
    logical(c_bool), device, target, contiguous, intent(inout) :: output(:, :, :, :, :)
    type(fortonnx_status), intent(inout) :: status
    call runtime_bind_tensor_pair(self, name, fortonnx_device_tensor(input), &
        fortonnx_device_tensor(output), status)
  end subroutine runtime_bind_device_5d_bool_bool

  subroutine runtime_bind_device_5d_bool_float64(self, name, input, output, status)
    class(fortonnx_runtime), intent(inout) :: self
    character(len=*), intent(in) :: name
    logical(c_bool), device, target, contiguous, intent(inout) :: input(:, :, :, :, :)
    real(c_double), device, target, contiguous, intent(inout) :: output(:, :, :, :, :)
    type(fortonnx_status), intent(inout) :: status
    call runtime_bind_tensor_pair(self, name, fortonnx_device_tensor(input), &
        fortonnx_device_tensor(output), status)
  end subroutine runtime_bind_device_5d_bool_float64

  subroutine runtime_bind_device_5d_bool_float32(self, name, input, output, status)
    class(fortonnx_runtime), intent(inout) :: self
    character(len=*), intent(in) :: name
    logical(c_bool), device, target, contiguous, intent(inout) :: input(:, :, :, :, :)
    real(c_float), device, target, contiguous, intent(inout) :: output(:, :, :, :, :)
    type(fortonnx_status), intent(inout) :: status
    call runtime_bind_tensor_pair(self, name, fortonnx_device_tensor(input), &
        fortonnx_device_tensor(output), status)
  end subroutine runtime_bind_device_5d_bool_float32

  subroutine runtime_bind_device_5d_float64_bool(self, name, input, output, status)
    class(fortonnx_runtime), intent(inout) :: self
    character(len=*), intent(in) :: name
    real(c_double), device, target, contiguous, intent(inout) :: input(:, :, :, :, :)
    logical(c_bool), device, target, contiguous, intent(inout) :: output(:, :, :, :, :)
    type(fortonnx_status), intent(inout) :: status
    call runtime_bind_tensor_pair(self, name, fortonnx_device_tensor(input), &
        fortonnx_device_tensor(output), status)
  end subroutine runtime_bind_device_5d_float64_bool

  subroutine runtime_bind_device_5d_float64_float64(self, name, input, output, status)
    class(fortonnx_runtime), intent(inout) :: self
    character(len=*), intent(in) :: name
    real(c_double), device, target, contiguous, intent(inout) :: input(:, :, :, :, :)
    real(c_double), device, target, contiguous, intent(inout) :: output(:, :, :, :, :)
    type(fortonnx_status), intent(inout) :: status
    call runtime_bind_tensor_pair(self, name, fortonnx_device_tensor(input), &
        fortonnx_device_tensor(output), status)
  end subroutine runtime_bind_device_5d_float64_float64

  subroutine runtime_bind_device_5d_float64_float32(self, name, input, output, status)
    class(fortonnx_runtime), intent(inout) :: self
    character(len=*), intent(in) :: name
    real(c_double), device, target, contiguous, intent(inout) :: input(:, :, :, :, :)
    real(c_float), device, target, contiguous, intent(inout) :: output(:, :, :, :, :)
    type(fortonnx_status), intent(inout) :: status
    call runtime_bind_tensor_pair(self, name, fortonnx_device_tensor(input), &
        fortonnx_device_tensor(output), status)
  end subroutine runtime_bind_device_5d_float64_float32

  subroutine runtime_bind_device_5d_float32_bool(self, name, input, output, status)
    class(fortonnx_runtime), intent(inout) :: self
    character(len=*), intent(in) :: name
    real(c_float), device, target, contiguous, intent(inout) :: input(:, :, :, :, :)
    logical(c_bool), device, target, contiguous, intent(inout) :: output(:, :, :, :, :)
    type(fortonnx_status), intent(inout) :: status
    call runtime_bind_tensor_pair(self, name, fortonnx_device_tensor(input), &
        fortonnx_device_tensor(output), status)
  end subroutine runtime_bind_device_5d_float32_bool

  subroutine runtime_bind_device_5d_float32_float64(self, name, input, output, status)
    class(fortonnx_runtime), intent(inout) :: self
    character(len=*), intent(in) :: name
    real(c_float), device, target, contiguous, intent(inout) :: input(:, :, :, :, :)
    real(c_double), device, target, contiguous, intent(inout) :: output(:, :, :, :, :)
    type(fortonnx_status), intent(inout) :: status
    call runtime_bind_tensor_pair(self, name, fortonnx_device_tensor(input), &
        fortonnx_device_tensor(output), status)
  end subroutine runtime_bind_device_5d_float32_float64

  subroutine runtime_bind_device_4d_bool_bool(self, name, input, output, status)
    class(fortonnx_runtime), intent(inout) :: self
    character(len=*), intent(in) :: name
    logical(c_bool), device, target, contiguous, intent(inout) :: input(:, :, :, :)
    logical(c_bool), device, target, contiguous, intent(inout) :: output(:, :, :, :)
    type(fortonnx_status), intent(inout) :: status
    call runtime_bind_tensor_pair(self, name, fortonnx_device_tensor(input), &
        fortonnx_device_tensor(output), status)
  end subroutine runtime_bind_device_4d_bool_bool

  subroutine runtime_bind_device_4d_bool_float64(self, name, input, output, status)
    class(fortonnx_runtime), intent(inout) :: self
    character(len=*), intent(in) :: name
    logical(c_bool), device, target, contiguous, intent(inout) :: input(:, :, :, :)
    real(c_double), device, target, contiguous, intent(inout) :: output(:, :, :, :)
    type(fortonnx_status), intent(inout) :: status
    call runtime_bind_tensor_pair(self, name, fortonnx_device_tensor(input), &
        fortonnx_device_tensor(output), status)
  end subroutine runtime_bind_device_4d_bool_float64

  subroutine runtime_bind_device_4d_bool_float32(self, name, input, output, status)
    class(fortonnx_runtime), intent(inout) :: self
    character(len=*), intent(in) :: name
    logical(c_bool), device, target, contiguous, intent(inout) :: input(:, :, :, :)
    real(c_float), device, target, contiguous, intent(inout) :: output(:, :, :, :)
    type(fortonnx_status), intent(inout) :: status
    call runtime_bind_tensor_pair(self, name, fortonnx_device_tensor(input), &
        fortonnx_device_tensor(output), status)
  end subroutine runtime_bind_device_4d_bool_float32

  subroutine runtime_bind_device_4d_float64_bool(self, name, input, output, status)
    class(fortonnx_runtime), intent(inout) :: self
    character(len=*), intent(in) :: name
    real(c_double), device, target, contiguous, intent(inout) :: input(:, :, :, :)
    logical(c_bool), device, target, contiguous, intent(inout) :: output(:, :, :, :)
    type(fortonnx_status), intent(inout) :: status
    call runtime_bind_tensor_pair(self, name, fortonnx_device_tensor(input), &
        fortonnx_device_tensor(output), status)
  end subroutine runtime_bind_device_4d_float64_bool

  subroutine runtime_bind_device_4d_float64_float64(self, name, input, output, status)
    class(fortonnx_runtime), intent(inout) :: self
    character(len=*), intent(in) :: name
    real(c_double), device, target, contiguous, intent(inout) :: input(:, :, :, :)
    real(c_double), device, target, contiguous, intent(inout) :: output(:, :, :, :)
    type(fortonnx_status), intent(inout) :: status
    call runtime_bind_tensor_pair(self, name, fortonnx_device_tensor(input), &
        fortonnx_device_tensor(output), status)
  end subroutine runtime_bind_device_4d_float64_float64

  subroutine runtime_bind_device_4d_float64_float32(self, name, input, output, status)
    class(fortonnx_runtime), intent(inout) :: self
    character(len=*), intent(in) :: name
    real(c_double), device, target, contiguous, intent(inout) :: input(:, :, :, :)
    real(c_float), device, target, contiguous, intent(inout) :: output(:, :, :, :)
    type(fortonnx_status), intent(inout) :: status
    call runtime_bind_tensor_pair(self, name, fortonnx_device_tensor(input), &
        fortonnx_device_tensor(output), status)
  end subroutine runtime_bind_device_4d_float64_float32

  subroutine runtime_bind_device_4d_float32_bool(self, name, input, output, status)
    class(fortonnx_runtime), intent(inout) :: self
    character(len=*), intent(in) :: name
    real(c_float), device, target, contiguous, intent(inout) :: input(:, :, :, :)
    logical(c_bool), device, target, contiguous, intent(inout) :: output(:, :, :, :)
    type(fortonnx_status), intent(inout) :: status
    call runtime_bind_tensor_pair(self, name, fortonnx_device_tensor(input), &
        fortonnx_device_tensor(output), status)
  end subroutine runtime_bind_device_4d_float32_bool

  subroutine runtime_bind_device_4d_float32_float64(self, name, input, output, status)
    class(fortonnx_runtime), intent(inout) :: self
    character(len=*), intent(in) :: name
    real(c_float), device, target, contiguous, intent(inout) :: input(:, :, :, :)
    real(c_double), device, target, contiguous, intent(inout) :: output(:, :, :, :)
    type(fortonnx_status), intent(inout) :: status
    call runtime_bind_tensor_pair(self, name, fortonnx_device_tensor(input), &
        fortonnx_device_tensor(output), status)
  end subroutine runtime_bind_device_4d_float32_float64

  subroutine runtime_bind_device_3d_bool_bool(self, name, input, output, status)
    class(fortonnx_runtime), intent(inout) :: self
    character(len=*), intent(in) :: name
    logical(c_bool), device, target, contiguous, intent(inout) :: input(:, :, :)
    logical(c_bool), device, target, contiguous, intent(inout) :: output(:, :, :)
    type(fortonnx_status), intent(inout) :: status
    call runtime_bind_tensor_pair(self, name, fortonnx_device_tensor(input), &
        fortonnx_device_tensor(output), status)
  end subroutine runtime_bind_device_3d_bool_bool

  subroutine runtime_bind_device_3d_bool_float64(self, name, input, output, status)
    class(fortonnx_runtime), intent(inout) :: self
    character(len=*), intent(in) :: name
    logical(c_bool), device, target, contiguous, intent(inout) :: input(:, :, :)
    real(c_double), device, target, contiguous, intent(inout) :: output(:, :, :)
    type(fortonnx_status), intent(inout) :: status
    call runtime_bind_tensor_pair(self, name, fortonnx_device_tensor(input), &
        fortonnx_device_tensor(output), status)
  end subroutine runtime_bind_device_3d_bool_float64

  subroutine runtime_bind_device_3d_bool_float32(self, name, input, output, status)
    class(fortonnx_runtime), intent(inout) :: self
    character(len=*), intent(in) :: name
    logical(c_bool), device, target, contiguous, intent(inout) :: input(:, :, :)
    real(c_float), device, target, contiguous, intent(inout) :: output(:, :, :)
    type(fortonnx_status), intent(inout) :: status
    call runtime_bind_tensor_pair(self, name, fortonnx_device_tensor(input), &
        fortonnx_device_tensor(output), status)
  end subroutine runtime_bind_device_3d_bool_float32

  subroutine runtime_bind_device_3d_float64_bool(self, name, input, output, status)
    class(fortonnx_runtime), intent(inout) :: self
    character(len=*), intent(in) :: name
    real(c_double), device, target, contiguous, intent(inout) :: input(:, :, :)
    logical(c_bool), device, target, contiguous, intent(inout) :: output(:, :, :)
    type(fortonnx_status), intent(inout) :: status
    call runtime_bind_tensor_pair(self, name, fortonnx_device_tensor(input), &
        fortonnx_device_tensor(output), status)
  end subroutine runtime_bind_device_3d_float64_bool

  subroutine runtime_bind_device_3d_float64_float64(self, name, input, output, status)
    class(fortonnx_runtime), intent(inout) :: self
    character(len=*), intent(in) :: name
    real(c_double), device, target, contiguous, intent(inout) :: input(:, :, :)
    real(c_double), device, target, contiguous, intent(inout) :: output(:, :, :)
    type(fortonnx_status), intent(inout) :: status
    call runtime_bind_tensor_pair(self, name, fortonnx_device_tensor(input), &
        fortonnx_device_tensor(output), status)
  end subroutine runtime_bind_device_3d_float64_float64

  subroutine runtime_bind_device_3d_float64_float32(self, name, input, output, status)
    class(fortonnx_runtime), intent(inout) :: self
    character(len=*), intent(in) :: name
    real(c_double), device, target, contiguous, intent(inout) :: input(:, :, :)
    real(c_float), device, target, contiguous, intent(inout) :: output(:, :, :)
    type(fortonnx_status), intent(inout) :: status
    call runtime_bind_tensor_pair(self, name, fortonnx_device_tensor(input), &
        fortonnx_device_tensor(output), status)
  end subroutine runtime_bind_device_3d_float64_float32

  subroutine runtime_bind_device_3d_float32_bool(self, name, input, output, status)
    class(fortonnx_runtime), intent(inout) :: self
    character(len=*), intent(in) :: name
    real(c_float), device, target, contiguous, intent(inout) :: input(:, :, :)
    logical(c_bool), device, target, contiguous, intent(inout) :: output(:, :, :)
    type(fortonnx_status), intent(inout) :: status
    call runtime_bind_tensor_pair(self, name, fortonnx_device_tensor(input), &
        fortonnx_device_tensor(output), status)
  end subroutine runtime_bind_device_3d_float32_bool

  subroutine runtime_bind_device_3d_float32_float64(self, name, input, output, status)
    class(fortonnx_runtime), intent(inout) :: self
    character(len=*), intent(in) :: name
    real(c_float), device, target, contiguous, intent(inout) :: input(:, :, :)
    real(c_double), device, target, contiguous, intent(inout) :: output(:, :, :)
    type(fortonnx_status), intent(inout) :: status
    call runtime_bind_tensor_pair(self, name, fortonnx_device_tensor(input), &
        fortonnx_device_tensor(output), status)
  end subroutine runtime_bind_device_3d_float32_float64

  subroutine runtime_bind_device_2d_bool_bool(self, name, input, output, status)
    class(fortonnx_runtime), intent(inout) :: self
    character(len=*), intent(in) :: name
    logical(c_bool), device, target, contiguous, intent(inout) :: input(:, :)
    logical(c_bool), device, target, contiguous, intent(inout) :: output(:, :)
    type(fortonnx_status), intent(inout) :: status
    call runtime_bind_tensor_pair(self, name, fortonnx_device_tensor(input), &
        fortonnx_device_tensor(output), status)
  end subroutine runtime_bind_device_2d_bool_bool

  subroutine runtime_bind_device_2d_bool_float64(self, name, input, output, status)
    class(fortonnx_runtime), intent(inout) :: self
    character(len=*), intent(in) :: name
    logical(c_bool), device, target, contiguous, intent(inout) :: input(:, :)
    real(c_double), device, target, contiguous, intent(inout) :: output(:, :)
    type(fortonnx_status), intent(inout) :: status
    call runtime_bind_tensor_pair(self, name, fortonnx_device_tensor(input), &
        fortonnx_device_tensor(output), status)
  end subroutine runtime_bind_device_2d_bool_float64

  subroutine runtime_bind_device_2d_bool_float32(self, name, input, output, status)
    class(fortonnx_runtime), intent(inout) :: self
    character(len=*), intent(in) :: name
    logical(c_bool), device, target, contiguous, intent(inout) :: input(:, :)
    real(c_float), device, target, contiguous, intent(inout) :: output(:, :)
    type(fortonnx_status), intent(inout) :: status
    call runtime_bind_tensor_pair(self, name, fortonnx_device_tensor(input), &
        fortonnx_device_tensor(output), status)
  end subroutine runtime_bind_device_2d_bool_float32

  subroutine runtime_bind_device_2d_float64_bool(self, name, input, output, status)
    class(fortonnx_runtime), intent(inout) :: self
    character(len=*), intent(in) :: name
    real(c_double), device, target, contiguous, intent(inout) :: input(:, :)
    logical(c_bool), device, target, contiguous, intent(inout) :: output(:, :)
    type(fortonnx_status), intent(inout) :: status
    call runtime_bind_tensor_pair(self, name, fortonnx_device_tensor(input), &
        fortonnx_device_tensor(output), status)
  end subroutine runtime_bind_device_2d_float64_bool

  subroutine runtime_bind_device_2d_float64_float64(self, name, input, output, status)
    class(fortonnx_runtime), intent(inout) :: self
    character(len=*), intent(in) :: name
    real(c_double), device, target, contiguous, intent(inout) :: input(:, :)
    real(c_double), device, target, contiguous, intent(inout) :: output(:, :)
    type(fortonnx_status), intent(inout) :: status
    call runtime_bind_tensor_pair(self, name, fortonnx_device_tensor(input), &
        fortonnx_device_tensor(output), status)
  end subroutine runtime_bind_device_2d_float64_float64

  subroutine runtime_bind_device_2d_float64_float32(self, name, input, output, status)
    class(fortonnx_runtime), intent(inout) :: self
    character(len=*), intent(in) :: name
    real(c_double), device, target, contiguous, intent(inout) :: input(:, :)
    real(c_float), device, target, contiguous, intent(inout) :: output(:, :)
    type(fortonnx_status), intent(inout) :: status
    call runtime_bind_tensor_pair(self, name, fortonnx_device_tensor(input), &
        fortonnx_device_tensor(output), status)
  end subroutine runtime_bind_device_2d_float64_float32

  subroutine runtime_bind_device_2d_float32_bool(self, name, input, output, status)
    class(fortonnx_runtime), intent(inout) :: self
    character(len=*), intent(in) :: name
    real(c_float), device, target, contiguous, intent(inout) :: input(:, :)
    logical(c_bool), device, target, contiguous, intent(inout) :: output(:, :)
    type(fortonnx_status), intent(inout) :: status
    call runtime_bind_tensor_pair(self, name, fortonnx_device_tensor(input), &
        fortonnx_device_tensor(output), status)
  end subroutine runtime_bind_device_2d_float32_bool

  subroutine runtime_bind_device_2d_float32_float64(self, name, input, output, status)
    class(fortonnx_runtime), intent(inout) :: self
    character(len=*), intent(in) :: name
    real(c_float), device, target, contiguous, intent(inout) :: input(:, :)
    real(c_double), device, target, contiguous, intent(inout) :: output(:, :)
    type(fortonnx_status), intent(inout) :: status
    call runtime_bind_tensor_pair(self, name, fortonnx_device_tensor(input), &
        fortonnx_device_tensor(output), status)
  end subroutine runtime_bind_device_2d_float32_float64

  subroutine runtime_bind_device_1d_bool_bool(self, name, input, output, status)
    class(fortonnx_runtime), intent(inout) :: self
    character(len=*), intent(in) :: name
    logical(c_bool), device, target, contiguous, intent(inout) :: input(:)
    logical(c_bool), device, target, contiguous, intent(inout) :: output(:)
    type(fortonnx_status), intent(inout) :: status
    call runtime_bind_tensor_pair(self, name, fortonnx_device_tensor(input), &
        fortonnx_device_tensor(output), status)
  end subroutine runtime_bind_device_1d_bool_bool

  subroutine runtime_bind_device_1d_bool_float64(self, name, input, output, status)
    class(fortonnx_runtime), intent(inout) :: self
    character(len=*), intent(in) :: name
    logical(c_bool), device, target, contiguous, intent(inout) :: input(:)
    real(c_double), device, target, contiguous, intent(inout) :: output(:)
    type(fortonnx_status), intent(inout) :: status
    call runtime_bind_tensor_pair(self, name, fortonnx_device_tensor(input), &
        fortonnx_device_tensor(output), status)
  end subroutine runtime_bind_device_1d_bool_float64

  subroutine runtime_bind_device_1d_bool_float32(self, name, input, output, status)
    class(fortonnx_runtime), intent(inout) :: self
    character(len=*), intent(in) :: name
    logical(c_bool), device, target, contiguous, intent(inout) :: input(:)
    real(c_float), device, target, contiguous, intent(inout) :: output(:)
    type(fortonnx_status), intent(inout) :: status
    call runtime_bind_tensor_pair(self, name, fortonnx_device_tensor(input), &
        fortonnx_device_tensor(output), status)
  end subroutine runtime_bind_device_1d_bool_float32

  subroutine runtime_bind_device_1d_float64_bool(self, name, input, output, status)
    class(fortonnx_runtime), intent(inout) :: self
    character(len=*), intent(in) :: name
    real(c_double), device, target, contiguous, intent(inout) :: input(:)
    logical(c_bool), device, target, contiguous, intent(inout) :: output(:)
    type(fortonnx_status), intent(inout) :: status
    call runtime_bind_tensor_pair(self, name, fortonnx_device_tensor(input), &
        fortonnx_device_tensor(output), status)
  end subroutine runtime_bind_device_1d_float64_bool

  subroutine runtime_bind_device_1d_float64_float64(self, name, input, output, status)
    class(fortonnx_runtime), intent(inout) :: self
    character(len=*), intent(in) :: name
    real(c_double), device, target, contiguous, intent(inout) :: input(:)
    real(c_double), device, target, contiguous, intent(inout) :: output(:)
    type(fortonnx_status), intent(inout) :: status
    call runtime_bind_tensor_pair(self, name, fortonnx_device_tensor(input), &
        fortonnx_device_tensor(output), status)
  end subroutine runtime_bind_device_1d_float64_float64

  subroutine runtime_bind_device_1d_float64_float32(self, name, input, output, status)
    class(fortonnx_runtime), intent(inout) :: self
    character(len=*), intent(in) :: name
    real(c_double), device, target, contiguous, intent(inout) :: input(:)
    real(c_float), device, target, contiguous, intent(inout) :: output(:)
    type(fortonnx_status), intent(inout) :: status
    call runtime_bind_tensor_pair(self, name, fortonnx_device_tensor(input), &
        fortonnx_device_tensor(output), status)
  end subroutine runtime_bind_device_1d_float64_float32

  subroutine runtime_bind_device_1d_float32_bool(self, name, input, output, status)
    class(fortonnx_runtime), intent(inout) :: self
    character(len=*), intent(in) :: name
    real(c_float), device, target, contiguous, intent(inout) :: input(:)
    logical(c_bool), device, target, contiguous, intent(inout) :: output(:)
    type(fortonnx_status), intent(inout) :: status
    call runtime_bind_tensor_pair(self, name, fortonnx_device_tensor(input), &
        fortonnx_device_tensor(output), status)
  end subroutine runtime_bind_device_1d_float32_bool

  subroutine runtime_bind_device_1d_float32_float64(self, name, input, output, status)
    class(fortonnx_runtime), intent(inout) :: self
    character(len=*), intent(in) :: name
    real(c_float), device, target, contiguous, intent(inout) :: input(:)
    real(c_double), device, target, contiguous, intent(inout) :: output(:)
    type(fortonnx_status), intent(inout) :: status
    call runtime_bind_tensor_pair(self, name, fortonnx_device_tensor(input), &
        fortonnx_device_tensor(output), status)
  end subroutine runtime_bind_device_1d_float32_float64
#endif

  subroutine runtime_run(self, name, status)
    class(fortonnx_runtime), intent(inout) :: self
    character(len=*), intent(in) :: name
    type(fortonnx_status), intent(inout) :: status
    integer(c_int) :: code
    integer :: index

    call status%clear()
    index = self%find(name)
    if (index == 0) then
      call set_status(status, fortonnx_invalid_argument, 'unknown model: ' // trim(name))
      return
    end if
    if (.not. self%models(index)%bound) then
      call set_status(status, fortonnx_invalid_argument, 'model buffers are not bound: ' // trim(name))
      return
    end if
    code = c_fortonnx_session_run(self%models(index)%session)
    if (code /= 0_c_int) then
      call set_c_error(status, 'inference failed for ' // trim(name), self%models(index)%session)
    end if
  end subroutine runtime_run

  subroutine runtime_run_all(self, status)
    class(fortonnx_runtime), intent(inout) :: self
    type(fortonnx_status), intent(inout) :: status
    integer :: index

    call status%clear()
    if (.not. self%initialized) then
      call set_status(status, fortonnx_invalid_argument, 'runtime is not initialized')
      return
    end if
    do index = 1, size(self%models)
      if (.not. c_associated(self%models(index)%session)) cycle
      if (.not. self%models(index)%bound) then
        call set_status(status, fortonnx_invalid_argument, &
            'model buffers are not bound: ' // self%models(index)%name)
        return
      end if
      if (c_fortonnx_session_run(self%models(index)%session) /= 0_c_int) then
        call set_c_error(status, 'inference failed for ' // self%models(index)%name, &
            self%models(index)%session)
        return
      end if
    end do
  end subroutine runtime_run_all

  subroutine runtime_features(self, name, input_features, output_features, status)
    class(fortonnx_runtime), intent(in) :: self
    character(len=*), intent(in) :: name
    integer(c_int64_t), intent(out) :: input_features, output_features
    type(fortonnx_status), intent(inout) :: status
    integer :: index

    call status%clear()
    input_features = 0_c_int64_t
    output_features = 0_c_int64_t
    index = self%find(name)
    if (index == 0) then
      call set_status(status, fortonnx_invalid_argument, 'unknown model: ' // trim(name))
      return
    end if
    if (size(self%models(index)%inputs) /= 1 .or. &
        size(self%models(index)%outputs) /= 1 .or. &
        size(self%models(index)%inputs(1)%shape) /= 2 .or. &
        size(self%models(index)%outputs(1)%shape) /= 2 .or. &
        self%models(index)%inputs(1)%shape(2) <= 0 .or. &
        self%models(index)%outputs(1)%shape(2) <= 0) then
      call set_status(status, fortonnx_invalid_argument, &
          'features is available only for rank-2 tensors with static feature dimensions')
      return
    end if
    input_features = self%models(index)%inputs(1)%shape(2)
    output_features = self%models(index)%outputs(1)%shape(2)
  end subroutine runtime_features

  subroutine runtime_shapes(self, name, input_shape, output_shape, status)
    class(fortonnx_runtime), intent(in) :: self
    character(len=*), intent(in) :: name
    integer(c_int64_t), allocatable, intent(out) :: input_shape(:), output_shape(:)
    type(fortonnx_status), intent(inout) :: status
    integer :: index

    call status%clear()
    index = self%find(name)
    if (index == 0) then
      call set_status(status, fortonnx_invalid_argument, 'unknown model: ' // trim(name))
      return
    end if
    if (size(self%models(index)%inputs) /= 1 .or. &
        size(self%models(index)%outputs) /= 1) then
      call set_status(status, fortonnx_invalid_argument, &
          'shapes is available only for models with one input and one output')
      return
    end if
    input_shape = self%models(index)%inputs(1)%shape
    output_shape = self%models(index)%outputs(1)%shape
  end subroutine runtime_shapes

  subroutine runtime_io_counts(self, name, input_count, output_count, status)
    class(fortonnx_runtime), intent(in) :: self
    character(len=*), intent(in) :: name
    integer, intent(out) :: input_count, output_count
    type(fortonnx_status), intent(inout) :: status
    integer :: index

    call status%clear()
    input_count = 0
    output_count = 0
    index = self%find(name)
    if (index == 0) then
      call set_status(status, fortonnx_invalid_argument, 'unknown model: ' // trim(name))
      return
    end if
    input_count = size(self%models(index)%inputs)
    output_count = size(self%models(index)%outputs)
  end subroutine runtime_io_counts

  subroutine runtime_tensor_shape(self, name, is_input, tensor_index, tensor_shape, status)
    class(fortonnx_runtime), intent(in) :: self
    character(len=*), intent(in) :: name
    integer(c_int), intent(in) :: is_input
    integer, intent(in) :: tensor_index
    integer(c_int64_t), allocatable, intent(out) :: tensor_shape(:)
    type(fortonnx_status), intent(inout) :: status
    integer :: index, tensor_count

    call status%clear()
    index = self%find(name)
    if (index == 0) then
      call set_status(status, fortonnx_invalid_argument, 'unknown model: ' // trim(name))
      return
    end if
    if (is_input == fortonnx_input) then
      tensor_count = size(self%models(index)%inputs)
    else if (is_input == fortonnx_output) then
      tensor_count = size(self%models(index)%outputs)
    else
      call set_status(status, fortonnx_invalid_argument, &
          'tensor kind must be fortonnx_input or fortonnx_output')
      return
    end if
    if (tensor_index < 1 .or. tensor_index > tensor_count) then
      call set_status(status, fortonnx_invalid_argument, 'tensor index is out of range')
      return
    end if
    if (is_input == fortonnx_input) then
      tensor_shape = self%models(index)%inputs(tensor_index)%shape
    else
      tensor_shape = self%models(index)%outputs(tensor_index)%shape
    end if
  end subroutine runtime_tensor_shape

  subroutine runtime_tensor_type(self, name, is_input, tensor_index, element_type, status)
    class(fortonnx_runtime), intent(in) :: self
    character(len=*), intent(in) :: name
    integer(c_int), intent(in) :: is_input
    integer, intent(in) :: tensor_index
    integer(c_int), intent(out) :: element_type
    type(fortonnx_status), intent(inout) :: status
    integer(c_int64_t), allocatable :: checked_shape(:)
    integer(c_int) :: code
    integer :: index

    element_type = 0_c_int
    call runtime_tensor_shape(self, name, is_input, tensor_index, checked_shape, status)
    if (.not. status%ok()) return
    index = self%find(name)
    code = c_fortonnx_session_get_tensor_type_at(self%models(index)%session, &
        is_input, int(tensor_index - 1, c_int64_t), element_type)
    if (code /= 0_c_int) call set_c_error(status, 'failed to inspect tensor type', self%models(index)%session)
  end subroutine runtime_tensor_type

  integer function runtime_find(self, name) result(index)
    class(fortonnx_runtime), intent(in) :: self
    character(len=*), intent(in) :: name
    integer :: candidate

    index = 0
    if (.not. allocated(self%models)) return
    do candidate = 1, size(self%models)
      if (.not. allocated(self%models(candidate)%name)) cycle
      if (self%models(candidate)%name == trim(name)) then
        index = candidate
        return
      end if
    end do
  end function runtime_find

  integer function runtime_size(self)
    class(fortonnx_runtime), intent(in) :: self
    if (allocated(self%models)) then
      runtime_size = size(self%models)
    else
      runtime_size = 0
    end if
  end function runtime_size

  logical function runtime_is_initialized(self)
    class(fortonnx_runtime), intent(in) :: self
    runtime_is_initialized = self%initialized
  end function runtime_is_initialized

  subroutine runtime_close(self)
    class(fortonnx_runtime), intent(inout) :: self
    integer :: index

    if (allocated(self%models)) then
      do index = 1, size(self%models)
        if (c_associated(self%models(index)%session)) then
          call c_fortonnx_session_destroy(self%models(index)%session)
          self%models(index)%session = c_null_ptr
        end if
      end do
      deallocate(self%models)
    end if
    if (c_associated(self%cpu_context)) then
      call c_fortonnx_cpu_context_destroy(self%cpu_context)
      self%cpu_context = c_null_ptr
    end if
    self%initialized = .false.
  end subroutine runtime_close

  subroutine runtime_finalize(self)
    type(fortonnx_runtime), intent(inout) :: self
    call self%close()
  end subroutine runtime_finalize
end module fortonnx
