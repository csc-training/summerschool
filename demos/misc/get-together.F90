program introductions
  implicit none
  character(len=:), dimension(:), allocatable :: names
  integer, dimension(:), allocatable :: order
  integer :: i, s


  names = [ &
       "Ali Afzalifar              ", &     
       "Amin Mirzai                ", &
       "Annabella Mondino Llermanos", &   
       "Arianna Toniato            ", &
       "Carlton Xavier             ", & 
       "Daragh Matthews            ", &   
       "Heidi Rytkönen             ", & 
       "Henrik Nortamo             ", &    
       "Irina Osadchuk             ", &
       "Joonas Nättilä             ", &   
       "Jonathan Velasco           ", & 
       "Jussi Enkovaara            ", &
       "Juho Jalava                ", & 
       "Kukka-Emilia Huhtinen      ", &
       "Lauri Tuppi                ", &            
       "Leevi Tuikka               ", &
       "Mahmoud Shepero            ", &  
       "Marko Kallio               ", &
       "Marta Lopez                ", &
       "Mehran Kiani               ", &         
       "Mika Jalava                ", &
       "Nadir Khan                 ", & 
       "Pekka Manninen             ", &         
       "Ruixue Feng                ", & 
       "Sami Ilvonen               ", &
       "Seija Sirkiä               ", &
       "Tomas Panoc                ", &
       "Tova Jarnerud              "  ]

  allocate (order(size(names)))
  call randomize_order(order)
  do i = 1, size(names)
     write(*,'(A1,I0,A1,I0,A1,A30)') '#', i, '/', &
       size(names), ':', names(order(i))
     read(*,*)
  end do

contains

  subroutine randomize_order(order)
    integer, intent(inout), dimension(:) :: order
    integer :: i, irand
    real :: r
    i = 1
    randloop: do while (i <= size(order))
       call random_number(r)
       r = real(size(order)-1) * r + 1.0
       irand = nint(r)
       if (all(order(1:i) /= irand)) then
          order(i) = irand
          i = i + 1
       else
          cycle randloop
       end if
    end do randloop
  end subroutine randomize_order
end program introductions

