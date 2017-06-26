! I/O routines for heat equation solver
module io
    use heat
    use mpi

contains

  ! Output routine, saves the temperature distribution as a png image
  ! Arguments:
  !   curr (type(field)): variable with the temperature data
  !   iter (integer): index of the time step
  subroutine write_field(curr, iter, parallel)

    use pngwriter
    implicit none
    type(field), intent(in) :: curr
    integer, intent(in) :: iter
    type(parallel_data), intent(in) :: parallel
    
    character(len=85) :: filename

    integer :: stat
    real(dp), dimension(:,:), allocatable, target :: full_data

    integer :: coords(2)
    integer :: ix, jy
    integer :: p, ierr

    if (parallel%rank == 0) then
       allocate(full_data(curr%nx_full, curr%ny_full))
       ! Copy rand #0 data to the global array
       full_data(1:curr%nx, 1:curr%ny) = curr%data(1:curr%nx, 1:curr%ny)

       ! Receive data from other ranks
       do p = 1, parallel%size - 1
          call mpi_cart_coords(parallel%comm, p, 2, coords, ierr)
          ix = coords(1) * curr%nx + 1
          jy = coords(2) * curr%ny + 1
          call mpi_recv(full_data(ix, jy), 1, parallel%subarraytype, p, 22, &
               & parallel%comm, MPI_STATUS_IGNORE, ierr)
       end do
       write(filename,'(A5,I4.4,A4,A)')  'heat_', iter, '.png'
       stat = save_png(full_data, curr%nx_full, curr%ny_full, filename)
       deallocate(full_data)
    else
       ! Send data
       call mpi_ssend(curr%data(1,1), 1, parallel%subarraytype, 0, 22, &
            & parallel%comm, ierr)
    end if

  end subroutine write_field


  ! Reads the temperature distribution from an input file
  ! Arguments:
  !   field0 (type(field)): field variable that will store the
  !                         read data
  !   filename (char): name of the input file
  ! Note that this version assumes the input data to be in C memory layout
  subroutine read_field(field0, filename, parallel)
    
    implicit none
    type(field), intent(out) :: field0
    character(len=85), intent(in) :: filename
    type(parallel_data), intent(out) :: parallel

    integer :: nx, ny, i, p, ierr
    character(len=2) :: dummy

    real(dp), dimension(:,:), allocatable :: full_data
    integer :: coords(2)
    integer :: ix, jy

    open(10, file=filename)
    ! Read the header
    read(10, *) dummy, nx, ny

    call parallel_setup(parallel, nx, ny)
    call set_field_dimensions(field0, nx, ny, parallel)

    ! The arrays for temperature field contain also a halo region
    allocate(field0%data(0:field0%nx+1, 0:field0%ny+1))

    if (parallel%rank == 0) then
       allocate(full_data(nx, ny))
       ! Read the data
       do i = 1, nx
          read(10, *) full_data(i, 1:ny)
       end do
       ! Copy own local part
       field0%data(1:field0%nx, 1:field0%ny) = full_data(1:field0%nx, 1:field0%ny)
       ! Send to other process
       do p=1, parallel%size - 1
          call mpi_cart_coords(parallel%comm, p, 2, coords, ierr)
          ix = coords(1) * field0%nx + 1
          jy = coords(2) * field0%ny + 1
          call mpi_send(full_data(ix, jy), 1, parallel%subarraytype, p, 44, &
               &   parallel%comm, ierr)
       end do
    else
       ! Receive data
       call mpi_recv(field0%data(1, 1), 1, parallel%subarraytype, 0, 44, &
            &        parallel%comm, MPI_STATUS_IGNORE, ierr)
    end if

    ! Set the boundary values
    field0%data(1:field0%nx, 0) = field0%data(1:field0%nx, 1)
    field0%data(1:field0%nx, field0%ny+1) = field0%data(1:field0%nx, field0%ny)
    field0%data(0, 0:field0%ny+1) = field0%data(1, 0:field0%ny+1)
    field0%data(field0%nx+1, 0:field0%ny+1) = field0%data(field0%nx, 0:field0%ny+1)

    close(10)
    if (parallel%rank == 0) then
       deallocate(full_data)
    end if

  end subroutine read_field
  
end module io
