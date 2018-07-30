/*
Lightweight Automated Planning Toolkit
Copyright (C) 2012
Miquel Ramirez <miquel.ramirez@rmit.edu.au>
Nir Lipovetzky <nirlipo@gmail.com>

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef __BIT_SQUARE_MATRIX__
#define __BIT_SQUARE_MATRIX__

#include "bit_array.hxx"

class Bit_Matrix
{
public:

	Bit_Matrix();
	
	Bit_Matrix( unsigned M, unsigned N );

	Bit_Matrix( Bit_Matrix& other );
	
	virtual ~Bit_Matrix();

	void set_dims( unsigned M, unsigned N );
	
	void set( unsigned i, unsigned j );
	
	void unset( unsigned i, unsigned j );

	unsigned iset( unsigned i, unsigned j );

	void clear() ;

	void resize( unsigned sz );
	
protected:
	Bit_Array*   m_data;
	unsigned     m_M;	//unsigned means unsigned int, 4 bytes.
	unsigned     m_N;
};

#endif // square_matrix.hxx
