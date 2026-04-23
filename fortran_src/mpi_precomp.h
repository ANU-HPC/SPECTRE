/*
 * SPDX-FileCopyrightText: 2026 École Polytechnique Fédérale de Lausanne - Swiss Plasma Center
 *
 * SPDX-License-Identifier: MIT
 */

! Define a set of preprocessor variables used to define MPI variables according to the correct interface.
! This is done to be backward compatible with MPI implementations that don't provide the mpi_f08 interface.
#ifdef NO_MPI_F08
#define USE_MPI use mpi
#define MPI_COMM_TYPE integer
#define MPI_REQUEST_TYPE integer
#define MPI_GROUP_TYPE integer
#define MPI_OP_TYPE integer
#define MPI_STATUS_TYPE integer, dimension(MPI_STATUS_SIZE)
#else
#define USE_MPI use mpi_f08
#define MPI_COMM_TYPE type(mpi_comm)
#define MPI_REQUEST_TYPE type(mpi_request)
#define MPI_GROUP_TYPE type(mpi_group)
#define MPI_OP_TYPE type(mpi_op)
#define MPI_STATUS_TYPE type(mpi_status)
#endif

