// kate: replace-tabs off; indent-width 4; indent-mode normal
// vim: ts=4:sw=4:noexpandtab
/*

Copyright (c) 2010--2012,
François Pomerleau and Stephane Magnenat, ASL, ETHZ, Switzerland
You can contact the authors at <f dot pomerleau at gmail dot com> and
<stephane at magnenat dot net>

All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of the <organization> nor the
      names of its contributors may be used to endorse or promote products
      derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL ETH-ASL BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

*/

#include <iostream>
#include <pointmatcher/PointMatcher.h>

#include "Eigen/QR"
#include "Eigen/Eigenvalues"
#include "Eigen/SVD"
#include "Eigen/Dense"

#include "ErrorMinimizersImpl.h"
#include "PointMatcherPrivate.h"
#include "Functions.h"

using namespace Eigen;
using namespace std;
using namespace PointMatcherSupport;

typedef PointMatcherSupport::Parametrizable Parametrizable;
typedef PointMatcherSupport::Parametrizable P;
typedef Parametrizable::Parameters Parameters;
typedef Parametrizable::ParameterDoc ParameterDoc;
typedef Parametrizable::ParametersDoc ParametersDoc;

template<typename T>
PointToPlaneWithPenaltiesErrorMinimizer<T>::PointToPlaneWithPenaltiesErrorMinimizer(const Parameters& params):
	PointToPlaneErrorMinimizer<T>(PointToPlaneWithPenaltiesErrorMinimizer::availableParameters(), params),
	confidenceInPenalties(Parametrizable::get<T>("confidenceInPenalties"))
{
}

template<typename T>
typename PointMatcher<T>::TransformationParameters PointToPlaneWithPenaltiesErrorMinimizer<T>::compute(const ErrorElements& mPts_const)
{
	ErrorElements mPts = mPts_const;
	const size_t dim(mPts_const.reference.features.rows() - 1);
	const size_t nbPenalty(mPts_const.penalties.size());
	const size_t nbPoints = mPts.weights.cols();



    mPts.weights = mPts.weights / mPts.weights.norm();
    mPts.weights.conservativeResize(Eigen::NoChange, nbPenalty * dim + nbPoints);


	Matrix location, cov, offset;

	//VK: This switches from Bab's penalties to Vlads penalty tests.

	if(nbPenalty){
		std::tie(location, cov, offset) = mPts_const.penalties[0];
		if(location.rows()==4){
			if(isnan(location(3,3))){
				typename PointMatcher<T>::TransformationParameters out =
						//PointToPlaneWithPenaltiesErrorMinimizer<T>::compute_with_gravity(mPts,location,cov);  // Penalty added by adding values to A in Ax=b
						PointToPlaneWithPenaltiesErrorMinimizer<T>::compute_4dof_with_gravity(mPts); // Computing in 4dof
				return out;
			}
		}
	}




	// It's hard to add points with descriptor to a Datapoints, so we create a new Datapoints for the new points and then concatenate it
	Matrix penaltiesPtsRead(dim + 1, nbPenalty * dim);
	Matrix penaltiesPtsReference(dim + 1, nbPenalty * dim);
	Matrix penaltiesNormals(dim, nbPenalty * dim);

	for (size_t i = 0; i < mPts_const.penalties.size(); ++i) {
		// To minimize both the distances from the point cloud and the penalties at the same time we convert the penalties to fake pairs of point/normal.
		// For each penalty n fake pairs of point/normal will be created, where n is the dimensions of the covariance.
		// The eigen decomposition of the penalty's covariance give us the following:
		// W: Covariance a n by n matrix
		// W = N * L * N^T
		// N = [n1 n2 n3]
		// where L is a diagonal matrix of the eigen value and N is a rotation matrix.
		// n1, n2, n3 are column vectors. The fake pairs will use these vectors as normal.
		// For the fake points of the reference and the reading the translation part of penalty tf matrix and the current transformation matrix will be used respectively.
		std::tie(location, cov, offset) = mPts_const.penalties[i];
		const Eigen::EigenSolver<Matrix> solver(cov);
		const Matrix eigenVec = solver.eigenvectors().real();
		const Vector eigenVal = solver.eigenvalues().real();
//		std::cout<< "Eigen Vector" << eigenVec << std::endl;
//		std::cout<< "Eigen Value" << eigenVal << std::endl;

		const Vector transInRef(location.col(dim));
		const Vector transInRead(mPts_const.T_refMean_iter * (offset).col(dim));

		penaltiesPtsRead.block(0, dim * i, dim + 1, dim) = transInRead.replicate(1, dim);
		penaltiesPtsReference.block(0, dim * i, dim + 1, dim) = transInRef.replicate(1, dim);
		penaltiesNormals.block(0, dim * i, dim, dim) = eigenVec;

		// The eigen value are the variance for each eigen vector.
		mPts.weights.block(0, nbPoints + dim * i, 1, dim) = eigenVal.diagonal().array().inverse().transpose();
//		std::cout<< "penaltiesNormals" << std::endl << penaltiesNormals << std::endl;
//		std::cout<< "penaltiesPtsRead" << std::endl << penaltiesPtsRead << std::endl;


	}
	const Labels normalLabel({Label("normals", dim)});
	const DataPoints penaltiesReference(penaltiesPtsReference, mPts_const.reference.featureLabels, penaltiesNormals, normalLabel);
	const DataPoints penaltiesRead(penaltiesPtsRead, mPts_const.reading.featureLabels);

	mPts.reference.concatenate(penaltiesReference);
	mPts.reading.concatenate(penaltiesRead);

//	std::cout<< "mPts.weights" << std::endl << mPts.weights << std::endl;
//	std::cout<< "mPts.reference" << std::endl << mPts.reference.features << std::endl;
//	std::cout<< "mPts.reading" << std::endl << mPts.reading.features << std::endl << std::endl;
//	std::cout<< "mPts.reference.descriptors" << std::endl << mPts.reference.descriptors << std::endl << std::endl;

	typename PointMatcher<T>::TransformationParameters out = PointToPlaneErrorMinimizer<T>::compute_in_place(mPts);

	return out;
}


template<typename T>
T PointToPlaneWithPenaltiesErrorMinimizer<T>::getResidualError(
				const DataPoints& filteredReading,
				const DataPoints& filteredReference,
				const OutlierWeights& outlierWeights,
				const Matches& matches,
				const Penalties& penalties,
				const TransformationParameters& T_refMean_iter) const
{
	assert(matches.ids.rows() > 0);

	// Fetch paired points
	typename ErrorMinimizer::ErrorElements mPts(filteredReading, filteredReference, outlierWeights, matches, penalties, T_refMean_iter);
	mPts.weights.row(0) = mPts.weights.row(0) / mPts.weights.row(0).norm();
	T pointToPlaneErr = PointToPlaneErrorMinimizer<T>::computeResidualError(mPts, false);

	// HACK FSR 2019
	T penalitiesErr = 0.0;
	for (const Penalty& p: penalties) {
		Vector e = T_refMean_iter.topRightCorner(3, 1) - std::get<0>(p).topRightCorner(3, 1);
		penalitiesErr += e.transpose() * std::get<1>(p).transpose() * e;
	}

	return pointToPlaneErr + penalitiesErr;
}


template<typename T>
typename PointMatcher<T>::TransformationParameters PointToPlaneWithPenaltiesErrorMinimizer<T>::compute_with_gravity(ErrorElements& mPts,
																													const Matrix& imu_attitude,
																													const Matrix& attitude_weight)
{
	const int dim = mPts.reading.features.rows();
	const int nbPts = mPts.reading.features.cols();




	if(this->force2D || dim == 3)
	{
		throw std::logic_error("compute_with_gravity() can be used solely in the 3D case!");
	}

	// Adjust if the user forces 2D minimization on XY-plane
	int forcedDim = dim - 1;
//	if(this->force2D && dim == 4)
//	{
//		mPts.reading.features.conservativeResize(3, Eigen::NoChange);
//		mPts.reading.features.row(2) = Matrix::Ones(1, nbPts);
//		mPts.reference.features.conservativeResize(3, Eigen::NoChange);
//		mPts.reference.features.row(2) = Matrix::Ones(1, nbPts);
//		forcedDim = dim - 2;
//	}

	// Fetch normal vectors of the reference point cloud (with adjustment if needed)
	const BOOST_AUTO(normalRef, mPts.reference.getDescriptorViewByName("normals").topRows(forcedDim));

	// Note: Normal vector must be precalculated to use this error. Use appropriate input filter.
	assert(normalRef.rows() > 0);

	// Compute cross product of cross = cross(reading X normalRef)
	const Matrix cross = this->crossProduct(mPts.reading.features, normalRef);

	// wF = [weights*cross, weights*normals]
	// F  = [cross, normals]
	Matrix wF(normalRef.rows()+ cross.rows(), normalRef.cols());
	Matrix F(normalRef.rows()+ cross.rows(), normalRef.cols());

	for(int i=0; i < cross.rows(); i++)
	{
		wF.row(i) = mPts.weights.array() * cross.row(i).array();
		F.row(i) = cross.row(i);
	}
	for(int i=0; i < normalRef.rows(); i++)
	{
		wF.row(i + cross.rows()) = mPts.weights.array() * normalRef.row(i).array();
		F.row(i + cross.rows()) = normalRef.row(i);
	}

	// Unadjust covariance A = wF * F'
	const Matrix A_orig = wF * F.transpose();

	const Matrix deltas = mPts.reading.features - mPts.reference.features;

	// dot product of dot = dot(deltas, normals)
	Matrix dotProd = Matrix::Zero(1, normalRef.cols());

	for(int i=0; i<normalRef.rows(); i++)
	{
		dotProd += (deltas.row(i).array() * normalRef.row(i).array()).matrix();
	}

	// b = -(wF' * dot)
	const Vector b_orig = -(wF * dotProd.transpose());


	//VK: Add extra entries to A and b that force IMU roll and pitch

	//attitude_weight
	Matrix A_grav(6,6);
    A_grav << 1, 0, 0, 0, 0, 0,
              0, 1, 0, 0, 0, 0,
              0, 0, 0, 0, 0, 0,
              0, 0, 0, 0, 0, 0,
              0, 0, 0, 0, 0, 0,
              0, 0, 0, 0, 0, 0;

	Vector b_grav = Matrix::Zero(6, 1);



    Matrix A = A_orig + A_grav*1000000;
    Vector b = b_orig + b_grav;


    std::cout << "This is A in Vlad's:" << std::endl << A << std::endl;
    std::cout << "This is b in Vlad's:" << std::endl << b << std::endl;

	Vector x(A.rows());

	solvePossiblyUnderdeterminedLinearSystem<T>(A, b, x);

	// Transform parameters to matrix
	Matrix mOut;
	if(dim == 4 && !this->force2D)
	{
		Eigen::Transform<T, 3, Eigen::Affine> transform;
		// PLEASE DONT USE EULAR ANGLES!!!!
		// Rotation in Eular angles follow roll-pitch-yaw (1-2-3) rule
		/*transform = Eigen::AngleAxis<T>(x(0), Eigen::Matrix<T,1,3>::UnitX())
         * Eigen::AngleAxis<T>(x(1), Eigen::Matrix<T,1,3>::UnitY())
         * Eigen::AngleAxis<T>(x(2), Eigen::Matrix<T,1,3>::UnitZ());*/

		transform = Eigen::AngleAxis<T>(x.head(3).norm(),x.head(3).normalized());

		// Reverse roll-pitch-yaw conversion, very useful piece of knowledge, keep it with you all time!
		/*const T pitch = -asin(transform(2,0));
            const T roll = atan2(transform(2,1), transform(2,2));
            const T yaw = atan2(transform(1,0) / cos(pitch), transform(0,0) / cos(pitch));
            std::cerr << "d angles" << x(0) - roll << ", " << x(1) - pitch << "," << x(2) - yaw << std::endl;*/
		transform.translation() = x.segment(3, 3);
		mOut = transform.matrix();

		if (mOut != mOut)
		{
			// Degenerate situation. This can happen when the source and reading clouds
			// are identical, and then b and x above are 0, and the rotation matrix cannot
			// be determined, it comes out full of NaNs. The correct rotation is the identity.
			mOut.block(0, 0, dim-1, dim-1) = Matrix::Identity(dim-1, dim-1);
		}
	}
	else
	{
		Eigen::Transform<T, 2, Eigen::Affine> transform;
		transform = Eigen::Rotation2D<T> (x(0));
		transform.translation() = x.segment(1, 2);

		if(this->force2D)
		{
			mOut = Matrix::Identity(dim, dim);
			mOut.topLeftCorner(2, 2) = transform.matrix().topLeftCorner(2, 2);
			mOut.topRightCorner(2, 1) = transform.matrix().topRightCorner(2, 1);
		}
		else
		{
			mOut = transform.matrix();
		}
	}
	return mOut;
}


//VK: This function will be removed or replaced, it is not used (imu attitude is not required, it is already in the prior input pointcloud)
template<typename T>
typename PointMatcher<T>::Matrix PointToPlaneWithPenaltiesErrorMinimizer<T>::compute_A_matrix_rows_for_gravity(const Matrix& imu_attitude, const Vector& normal_vect)
{
	Matrix A_grav_11 = Matrix::Zero(3, 3);
	T n1 = normal_vect(0);
	T n2 = normal_vect(1);
	T n3 = normal_vect(2);
	T r3 = imu_attitude(0,2);
	T r6 = imu_attitude(1,2);
	T r9 = imu_attitude(2,2);
	A_grav_11 << pow(n3*r6-n2*r9,2), -(n3*r3-n1*r9)*(n3*r6-n2*r9), (n2*r3-n1*r6)*(n3*r6-n2*r9),
	             -(n3*r3-n1*r9)*(n3*r6-n2*r9), pow((n3*r3-n1*r9),2), -(n2*r3-n1*r6)*(n3*r3-n1*r9),
			     (n2*r3-n1*r6)*(n3*r6-n2*r9), -(n2*r3-n1*r6)*(n3*r3-n1*r9), pow((n2*r3-n1*r6),2);

	Matrix A_grav = Matrix::Zero(6, 6);
	A_grav.block(0,0,3,3) = A_grav_11;
	return A_grav;
}

//VK: Similarly for this function, not needed either
template<typename T>
typename PointMatcher<T>::Vector PointToPlaneWithPenaltiesErrorMinimizer<T>::compute_b_vector_elements_for_gravity(const Matrix& imu_attitude, const Vector& normal_vect)
{
	Vector b_grav = Vector::Zero(6);
	T n1 = normal_vect(0);
	T n2 = normal_vect(1);
	T n3 = normal_vect(2);
	T r3 = imu_attitude(0,2);
	T r6 = imu_attitude(1,2);
	T r9 = imu_attitude(2,2);

	b_grav(0) = -(n3*r6 - n2*r9)*(n1*r3 - n3 + n2*r6 + n3*r9);
	b_grav(1) =  (n3*r3 - n1*r9)*(n1*r3 - n3 + n2*r6 + n3*r9);
	b_grav(2) = -(n2*r3 - n1*r6)*(n1*r3 - n3 + n2*r6 + n3*r9);

	return b_grav;
}


// This is a modified version of the standard point-to-plane minimizer, which however works in 4DOF (the reference pointcloud MUST be gravity oriented)
template<typename T>
typename PointMatcher<T>::TransformationParameters PointToPlaneWithPenaltiesErrorMinimizer<T>::compute_4dof_with_gravity(ErrorElements& mPts)
{
    const int dim = mPts.reading.features.rows();
    const int nbPts = mPts.reading.features.cols();




    if(this->force2D || dim == 3)
    {
        throw std::logic_error("compute_4dof_with_gravity() can be used solely in the 3D case!");
    }

    // Adjust if the user forces 2D minimization on XY-plane
    int forcedDim = dim - 1;
//	if(this->force2D && dim == 4)
//	{
//		mPts.reading.features.conservativeResize(3, Eigen::NoChange);
//		mPts.reading.features.row(2) = Matrix::Ones(1, nbPts);
//		mPts.reference.features.conservativeResize(3, Eigen::NoChange);
//		mPts.reference.features.row(2) = Matrix::Ones(1, nbPts);
//		forcedDim = dim - 2;
//	}

    // Fetch normal vectors of the reference point cloud (with adjustment if needed)
    const BOOST_AUTO(normalRef, mPts.reference.getDescriptorViewByName("normals").topRows(forcedDim));

    // Note: Normal vector must be precalculated to use this error. Use appropriate input filter.
    assert(normalRef.rows() > 0);



    //VK: The cross product is replaced by a modified vector scalar product (GAMMA*reading)dot(normal)
    Matrix Gamma(3,3);
	Gamma << 0,-1, 0,
	         1, 0, 0,
	         0, 0, 0;


    // Compute cross product of cross = cross(reading X normalRef)
    //const Matrix cross = this->crossProduct(mPts.reading.features, normalRef);             // This would be 3*k matrix
	const Matrix cross = ((Gamma*mPts.reading.features).transpose()*normalRef).diagonal().transpose();   // This is 1*k vector


    // wF = [weights*cross, weights*normals]
    // F  = [cross, normals]
    Matrix wF(normalRef.rows()+ cross.rows(), normalRef.cols());
    Matrix F(normalRef.rows()+ cross.rows(), normalRef.cols());

	std::cout << "wF rows:" << wF.rows() << " and cols:" << wF.cols()  << std::endl;
	std::cout << "F rows:" << F.rows() << " and cols:" << F.cols() << std::endl;

    for(int i=0; i < cross.rows(); i++)
    {
        wF.row(i) = mPts.weights.array() * cross.row(i).array();
        F.row(i) = cross.row(i);
    }
    for(int i=0; i < normalRef.rows(); i++)
    {
        wF.row(i + cross.rows()) = mPts.weights.array() * normalRef.row(i).array();
        F.row(i + cross.rows()) = normalRef.row(i);
    }

    // Unadjust covariance A = wF * F'
    const Matrix A = wF * F.transpose();

    const Matrix deltas = mPts.reading.features - mPts.reference.features;

    // dot product of dot = dot(deltas, normals)
    Matrix dotProd = Matrix::Zero(1, normalRef.cols());

    for(int i=0; i<normalRef.rows(); i++)
    {
        dotProd += (deltas.row(i).array() * normalRef.row(i).array()).matrix();
    }

    // b = -(wF' * dot)
    const Vector b = -(wF * dotProd.transpose());



    std::cout << "This is A in Vlad's 4dof:" << std::endl << A << std::endl;
    std::cout << "This is b in Vlad's 4dof:" << std::endl << b << std::endl;

    Vector x(A.rows());

    solvePossiblyUnderdeterminedLinearSystem<T>(A, b, x);

	std::cout << "This is x in Vlad's 4dof:" << std::endl << x << std::endl;

    // Transform parameters to matrix
    Matrix mOut;
    if(dim == 4 && !this->force2D)
    {
        Eigen::Transform<T, 3, Eigen::Affine> transform;
        // PLEASE DONT USE EULAR ANGLES!!!!
        // Rotation in Eular angles follow roll-pitch-yaw (1-2-3) rule
        /*transform = Eigen::AngleAxis<T>(x(0), Eigen::Matrix<T,1,3>::UnitX())
         * Eigen::AngleAxis<T>(x(1), Eigen::Matrix<T,1,3>::UnitY())
         * Eigen::AngleAxis<T>(x(2), Eigen::Matrix<T,1,3>::UnitZ());*/

        //transform = Eigen::AngleAxis<T>(x.head(3).norm(),x.head(3).normalized());
		Vector unitZ(3,1);
		unitZ << 0,0,1;
        transform = Eigen::AngleAxis<T>(x(0), unitZ);

        // Reverse roll-pitch-yaw conversion, very useful piece of knowledge, keep it with you all time!
        /*const T pitch = -asin(transform(2,0));
            const T roll = atan2(transform(2,1), transform(2,2));
            const T yaw = atan2(transform(1,0) / cos(pitch), transform(0,0) / cos(pitch));
            std::cerr << "d angles" << x(0) - roll << ", " << x(1) - pitch << "," << x(2) - yaw << std::endl;*/
        transform.translation() = x.segment(1, 3);
        mOut = transform.matrix();

        if (mOut != mOut)
        {
            // Degenerate situation. This can happen when the source and reading clouds
            // are identical, and then b and x above are 0, and the rotation matrix cannot
            // be determined, it comes out full of NaNs. The correct rotation is the identity.
            mOut.block(0, 0, dim-1, dim-1) = Matrix::Identity(dim-1, dim-1);
        }
    }
    else
    {
        Eigen::Transform<T, 2, Eigen::Affine> transform;
        transform = Eigen::Rotation2D<T> (x(0));
        transform.translation() = x.segment(1, 2);

        if(this->force2D)
        {
            mOut = Matrix::Identity(dim, dim);
            mOut.topLeftCorner(2, 2) = transform.matrix().topLeftCorner(2, 2);
            mOut.topRightCorner(2, 1) = transform.matrix().topRightCorner(2, 1);
        }
        else
        {
            mOut = transform.matrix();
        }
    }
    return mOut;
}

template struct PointToPlaneWithPenaltiesErrorMinimizer<float>;
template struct PointToPlaneWithPenaltiesErrorMinimizer<double>;
