# .bash_profile

# Get the aliases and functions
if [ -f ~/.bashrc ]; then
	. ~/.bashrc
fi

# User specific environment and startup programs

export PETSC_DIR=$HOME/local/src/petsc-3.4.3
export PETSC_ARCH=arch-linux2-fftw
export PETSC_LIB=$HOME/local/src/petsc-3.4.3/arch-linux2-fftw/lib
