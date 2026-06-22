/* -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*- */

/*
 Copyright (C) 2026 Colin Alberts

 This file is part of QuantLib, a free-software/open-source library
 for financial quantitative analysts and developers - http://quantlib.org/

 QuantLib is free software: you can redistribute it and/or modify it
 under the terms of the QuantLib license.  You should have received a
 copy of the license along with this program; if not, please email
 <quantlib-dev@lists.sf.net>. The license is also available online at
 <https://www.quantlib.org/license.shtml>.

 This program is distributed in the hope that it will be useful, but WITHOUT
 ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 FOR A PARTICULAR PURPOSE.  See the license for more details.
*/

/*! \file cmaesoptimizer.hpp
    \brief Covariance Matrix Adaptation Evolution Strategy (CMA-ES)
*/

#ifndef quantlib_optimization_cmaes_hpp
#define quantlib_optimization_cmaes_hpp

#include <ql/math/optimization/constraint.hpp>
#include <ql/math/optimization/problem.hpp>
#include <ql/math/randomnumbers/mt19937uniformrng.hpp>
#include <ql/math/randomnumbers/seedgenerator.hpp>
#include <ql/utilities/null.hpp>

namespace QuantLib {

    //! (mu/mu_w, lambda)-CMA-ES, a gradient-free global optimizer.
    /*! Each generation samples lambda points from N(m, sigma^2 C), ranks them by
        objective value, and recombines the best mu to update the mean m, the step
        size sigma (cumulative step-size adaptation), and the covariance C (rank-one
        plus rank-mu). Strategy parameters and update equations follow N. Hansen,
        "The CMA Evolution Strategy: A Tutorial," arXiv:1604.00772.

        The initial mean and step size are taken from Problem::currentValue() and
        \p sigma. Box bounds on the Problem's constraint are handled by repair
        (clipping for evaluation) plus a quadratic penalty on out-of-box samples;
        a coordinate whose bound is +/-DBL_MAX is treated as unbounded, so without
        finite bounds the method runs as unconstrained CMA-ES. The reported
        solution is always feasible.

        \ingroup optimizers
    */
    class CMAES : public OptimizationMethod {
      public:
        /*! \param lambda          offspring per generation; Null<Size>() -> 4 + floor(3 ln n)
            \param mu              selected parents; Null<Size>() -> floor(lambda/2)
            \param sigma           initial step size (e.g. ~0.3 of the box width when bounded)
            \param boundaryPenalty weight on the box-violation penalty (used only with bounds)
            \param seed            random number generator seed
        */
        explicit CMAES(Size lambda = Null<Size>(),
                       Size mu = Null<Size>(),
                       Real sigma = 0.3,
                       Real boundaryPenalty = 1.0,
                       unsigned long seed = SeedGenerator::instance().get());

        EndCriteria::Type minimize(Problem& P, const EndCriteria& endCriteria) override;

      private:
        Size lambda_, mu_; // requested; Null<Size>() => auto-set in minimize()
        Real sigma0_, boundaryPenalty_;
        MersenneTwisterUniformRng rng_;
    };

}

#endif
