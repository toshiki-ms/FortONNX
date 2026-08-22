program scientific_package_cpu
  use, intrinsic :: iso_c_binding, only: c_float
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
  use fortonnx
  implicit none

  integer, parameter :: batch_size = 3
  real(c_float), parameter :: absolute_tolerance = 1.0e-6_c_float
  real(c_float), parameter :: relative_tolerance = 1.0e-6_c_float
  type(fortonnx_runtime) :: runtime
  type(fortonnx_options) :: options
  type(fortonnx_status) :: status
  type(fortonnx_tensor) :: inputs(2), outputs(1)
  real(c_float), target :: temperature_anomaly(1, batch_size)
  real(c_float), target :: radiative_forcing(1, batch_size)
  real(c_float), target :: response(1, batch_size)
  real(c_float) :: expected(1, batch_size), allowed_error(1, batch_size)
  character(len=2048) :: package_root, model_path, verification_path
  character(len=512) :: header
  integer :: file_unit, io_status, sample

  if (command_argument_count() /= 1) then
    write(*, '(a)') 'usage: scientific_package_cpu PACKAGE_DIRECTORY'
    stop 2
  end if
  call get_command_argument(1, package_root)

  model_path = trim(package_root) // '/models/linear-response.onnx'
  verification_path = trim(package_root) // '/verification/nominal.csv'

  open(newunit=file_unit, file=trim(verification_path), status='old', &
      action='read', iostat=io_status)
  if (io_status /= 0) error stop 'cannot open verification/nominal.csv'
  read(file_unit, '(a)', iostat=io_status) header
  if (io_status /= 0) error stop 'cannot read verification CSV header'
  do sample = 1, batch_size
    read(file_unit, *, iostat=io_status) temperature_anomaly(1, sample), &
        radiative_forcing(1, sample), expected(1, sample)
    if (io_status /= 0) error stop 'invalid verification CSV row'
  end do
  close(file_unit)

  if (.not. all(ieee_is_finite(temperature_anomaly)) .or. &
      .not. all(ieee_is_finite(radiative_forcing))) then
    error stop 'verification inputs must be finite'
  end if
  if (any(temperature_anomaly < -100.0_c_float) .or. &
      any(temperature_anomaly > 100.0_c_float)) then
    error stop 'temperature anomaly is outside interface-domain'
  end if
  if (any(radiative_forcing < -2000.0_c_float) .or. &
      any(radiative_forcing > 2000.0_c_float)) then
    error stop 'radiative forcing is outside interface-domain'
  end if

  options%backend = fortonnx_cpu
  options%intra_op_threads = 2
  call runtime%initialize(options, status)
  call require_ok(status)
  call runtime%load('linear_response', trim(model_path), status)
  call require_ok(status)

  ! Deliberately bind the two inputs in reverse graph order. Exact ONNX names
  ! provide the mapping. ONNX [batch, 1] maps to Fortran (1, batch).
  inputs(1) = fortonnx_host_tensor(radiative_forcing, 'radiative_forcing')
  inputs(2) = fortonnx_host_tensor(temperature_anomaly, 'temperature_anomaly')
  outputs(1) = fortonnx_host_tensor(response, 'temperature_response')
  call runtime%bind('linear_response', inputs, outputs, status)
  call require_ok(status)
  call runtime%run('linear_response', status)
  call require_ok(status)

  if (.not. all(ieee_is_finite(response))) then
    error stop 'inference produced a non-finite response'
  end if
  allowed_error = absolute_tolerance + relative_tolerance * abs(expected)
  if (any(abs(response - expected) > allowed_error)) then
    write(*, '(a,3f12.6)') 'actual:   ', response
    write(*, '(a,3f12.6)') 'expected: ', expected
    error stop 'scientific package known-answer verification failed'
  end if

  call runtime%close()
  write(*, '(a,es12.4)') 'Scientific package verification passed; max error = ', &
      maxval(abs(response - expected))

contains

  subroutine require_ok(current)
    type(fortonnx_status), intent(in) :: current
    if (.not. current%ok()) then
      write(*, '(a,i0,2a)') 'FortONNX error ', current%code, ': ', current%message
      error stop 1
    end if
  end subroutine require_ok
end program scientific_package_cpu
