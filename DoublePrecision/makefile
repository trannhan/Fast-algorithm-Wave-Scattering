ALL:  scat
CFLAGS = 
FFLAGS =
CPPFLAGS      = 
FPPFLAGS      =
CLEANFILES  =  scat

include  ${PETSC_DIR}/conf/variables
include  ${PETSC_DIR}/conf/rules

scat:  scat.o  chkopts
	${CLINKER}  -o  scat  scat.o    ${PETSC_LIB}
	${RM}  scat.o
