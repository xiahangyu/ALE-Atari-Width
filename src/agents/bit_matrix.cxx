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

#include "bit_matrix.hxx"


	Bit_Matrix::Bit_Matrix()
		: m_data(NULL), m_M(0), m_N( 0 )
	{
	}
	
	Bit_Matrix::Bit_Matrix( unsigned M, unsigned N )
		: m_data(NULL), m_M(M), m_N( N )
	{
		resize( m_M*m_N );
	}

	Bit_Matrix::Bit_Matrix( Bit_Matrix& other )
	{
		m_N = other.m_N;
		m_M = other.m_M;
		m_data = new Bit_Array( *(other.m_data) );
	}
	
	Bit_Matrix::~Bit_Matrix()
	{
		if ( m_data ) delete m_data;
	}

	void Bit_Matrix::set_dims( unsigned M, unsigned N )
	{
		m_N = N;
		m_M = M;
		resize( m_M*m_N );
	}
	
	void Bit_Matrix::set( unsigned i, unsigned j )
	{
		m_data->set( i*m_N+j );
	}
	
	void Bit_Matrix::unset( unsigned i, unsigned j )
	{
		m_data->unset( i*m_N+j);
	}

	unsigned Bit_Matrix::iset( unsigned i, unsigned j )
	{
		return m_data->isset( i*m_N+j );
	}

	void Bit_Matrix::clear() 
	{
		m_data->reset();
	}	

	void Bit_Matrix::resize( unsigned sz )
	{
		m_data = new Bit_Array(sz);
		m_data->reset();
	}
	

