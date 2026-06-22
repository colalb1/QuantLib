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

#include <ql/experimental/math/cmaesoptimizer.hpp>
#include <ql/math/distributions/normaldistribution.hpp>
#include <ql/math/matrix.hpp>
#include <ql/math/matrixutilities/symmetricschurdecomposition.hpp>
#include <ql/math/randomnumbers/inversecumulativerng.hpp>
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <vector>

namespace QuantLib {

    CMAES::CMAES(Size lambda, Size mu, Real sigma, Real boundaryPenalty, unsigned long seed)
    : lambda_(lambda), mu_(mu), sigma0_(sigma), boundaryPenalty_(boundaryPenalty), rng_(seed) {
        QL_REQUIRE(sigma0_ > 0.0, "CMA-ES initial step size (sigma) must be positive");
        QL_REQUIRE(boundaryPenalty_ >= 0.0, "CMA-ES boundary penalty must be non-negative");
    }

    EndCriteria::Type CMAES::minimize(Problem& P, const EndCriteria& endCriteria) {
        EndCriteria::Type ecType = EndCriteria::None;
        P.reset();

        const Array x0 = P.currentValue();
        const Size n = x0.size();
        QL_REQUIRE(n > 0, "CMA-ES needs a non-empty initial guess");

        // default strategy parameters (Hansen, arXiv:1604.00772)
        const Size lambda =
            (lambda_ == Null<Size>()) ? Size(4 + std::floor(3.0 * std::log(Real(n)))) : lambda_;
        QL_REQUIRE(lambda >= 2, "CMA-ES population (lambda) must be at least 2");
        const Size mu = (mu_ == Null<Size>()) ? Size(lambda / 2) : mu_;
        QL_REQUIRE(mu >= 1 && mu <= lambda, "CMA-ES parents (mu) must satisfy 1 <= mu <= lambda");

        // recombination weights: positive, normalized to sum to one
        Array weights(mu);

        for (Size i = 0; i < mu; ++i)
            weights[i] = std::log(0.5 * (lambda + 1.0)) - std::log(Real(i + 1));
            
        weights /= std::accumulate(weights.begin(), weights.end(), Real(0.0));
        const Real muEff = 1.0 / DotProduct(weights, weights); // variance-effective selection mass

        const Real cSigma = (muEff + 2.0) / (n + muEff + 5.0);
        const Real dSigma =
            1.0 + 2.0 * std::max(0.0, std::sqrt((muEff - 1.0) / (n + 1.0)) - 1.0) + cSigma;
        const Real cc = (4.0 + muEff / n) / (n + 4.0 + 2.0 * muEff / n);
        const Real c1 = 2.0 / ((n + 1.3) * (n + 1.3) + muEff);
        const Real cmu =
            std::min(1.0 - c1, 2.0 * (muEff - 2.0 + 1.0 / muEff) / ((n + 2.0) * (n + 2.0) + muEff));
        const Real chiN = std::sqrt(Real(n)) * (1.0 - 1.0 / (4.0 * n) + 1.0 / (21.0 * n * n));

        // generations between lazy eigen-decompositions of C
        const Size lazyGens = std::max<Size>(1, Size(std::floor(1.0 / (10.0 * (c1 + cmu) * n))));

        // box bounds: a coordinate is bounded only if its bound is finite
        // (different from +/-DBL_MAX), otherwise it is unconstrained
        const Real maxReal = std::numeric_limits<Real>::max();
        Array lower(n, -maxReal), upper(n, maxReal);
        if (!P.constraint().empty()) {
            lower = P.constraint().lowerBound(x0);
            upper = P.constraint().upperBound(x0);
        }
        std::vector<bool> boundedLow(n), boundedHigh(n);
        bool hasBounds = false;
        for (Size j = 0; j < n; ++j) {
            boundedLow[j] = lower[j] > -0.5 * maxReal;
            boundedHigh[j] = upper[j] < 0.5 * maxReal;
            hasBounds = hasBounds || boundedLow[j] || boundedHigh[j];
        }
        auto repair = [&](Array x) {
            if (hasBounds) {
                for (Size j = 0; j < n; ++j) {
                    if (boundedLow[j] && x[j] < lower[j])
                        x[j] = lower[j];
                    if (boundedHigh[j] && x[j] > upper[j])
                        x[j] = upper[j];
                }
            }
            return x;
        };

        // dynamic state
        Array mean = x0;
        Real sigma = sigma0_;
        Matrix C(n, n, 0.0), B(n, n, 0.0), invSqrtC(n, n, 0.0);
        Array dDiag(n, 1.0); // square roots of the eigenvalues of C
        for (Size i = 0; i < n; ++i)
            C[i][i] = B[i][i] = invSqrtC[i][i] = 1.0;
        Array pSigma(n, 0.0), pCov(n, 0.0);

        InverseCumulativeRng<MersenneTwisterUniformRng, InverseCumulativeNormal> gauss(rng_);

        // best feasible point so far, seeded with the repaired initial guess
        Array bestX = repair(x0);
        Real bestF = P.value(bestX);

        Size statState = 0;
        Real fBestPrev = bestF;
        Size gen = 0;

        std::vector<Array> ys(lambda), xs(lambda), xFeas(lambda);
        Array fFeas(lambda), fvals(lambda);
        std::vector<Size> idx(lambda);

        do {
            ++gen;

            // sample offspring: x_k = mean + sigma * B * (D .* z_k)
            for (Size k = 0; k < lambda; ++k) {
                Array z(n);
                for (Size j = 0; j < n; ++j)
                    z[j] = gauss.next().value;
                ys[k] = B * (dDiag * z);
                xs[k] = mean + sigma * ys[k];
            }

            // evaluate: repair (clip) + quadratic penalty on box violations
            if (hasBounds) {
                Real lo = QL_MAX_REAL, hi = -QL_MAX_REAL;
                std::vector<Real> absF(lambda);
                for (Size k = 0; k < lambda; ++k) {
                    xFeas[k] = repair(xs[k]);
                    fFeas[k] = P.value(xFeas[k]);
                    absF[k] = std::fabs(fFeas[k]);
                    lo = std::min(lo, fFeas[k]);
                    hi = std::max(hi, fFeas[k]);
                }
                // objective scale that keeps the penalty commensurate with f
                std::nth_element(absF.begin(), absF.begin() + absF.size() / 2, absF.end());
                Real fScale = std::max(absF[absF.size() / 2], hi - lo);
                if (fScale <= 0.0)
                    fScale = 1.0;
                for (Size k = 0; k < lambda; ++k) {
                    Real penalty = 0.0;
                    for (Size j = 0; j < n; ++j) {
                        Real d = xs[k][j] - xFeas[k][j];
                        if (d != 0.0) {
                            Real varj = sigma * sigma * std::max(C[j][j], QL_EPSILON);
                            penalty += (fScale / varj) * d * d;
                        }
                    }
                    fvals[k] = fFeas[k] + boundaryPenalty_ * penalty;
                }
            } else {
                for (Size k = 0; k < lambda; ++k) {
                    xFeas[k] = xs[k];
                    fFeas[k] = P.value(xs[k]);
                    fvals[k] = fFeas[k];
                }
            }

            // rank by penalized objective value
            std::iota(idx.begin(), idx.end(), Size(0));
            std::sort(idx.begin(), idx.end(), [&](Size a, Size b) { return fvals[a] < fvals[b]; });

            // track the best feasible point by the true (unpenalized) objective
            Real fBestGen = QL_MAX_REAL;
            for (Size k = 0; k < lambda; ++k) {
                fBestGen = std::min(fBestGen, fFeas[k]);
                if (fFeas[k] < bestF) {
                    bestF = fFeas[k];
                    bestX = xFeas[k];
                }
            }

            // recombine the mu best (using the original, unclipped samples)
            mean = Array(n, 0.0);
            Array yw(n, 0.0);
            for (Size i = 0; i < mu; ++i) {
                mean += weights[i] * xs[idx[i]];
                yw += weights[i] * ys[idx[i]];
            }

            // step-size evolution path (CSA)
            pSigma = (1.0 - cSigma) * pSigma +
                     std::sqrt(cSigma * (2.0 - cSigma) * muEff) * (invSqrtC * yw);
            const Real psNorm = Norm2(pSigma);

            // Heaviside step: delays the covariance-path update while sigma adapts
            const Real hsig = (psNorm / std::sqrt(1.0 - std::pow(1.0 - cSigma, 2.0 * gen)) <
                               (1.4 + 2.0 / (n + 1.0)) * chiN) ?
                                  1.0 :
                                  0.0;

            // covariance evolution path
            pCov = (1.0 - cc) * pCov + hsig * std::sqrt(cc * (2.0 - cc) * muEff) * yw;

            // update step size
            sigma *= std::exp((cSigma / dSigma) * (psNorm / chiN - 1.0));

            // update covariance (rank-one + rank-mu)
            const Real deltaH = (1.0 - hsig) * cc * (2.0 - cc);
            Matrix rankMu(n, n, 0.0);
            for (Size i = 0; i < mu; ++i)
                rankMu += weights[i] * outerProduct(ys[idx[i]], ys[idx[i]]);
            C = (1.0 - c1 - cmu + c1 * deltaH) * C + c1 * outerProduct(pCov, pCov) + cmu * rankMu;

            // stopping criteria
            if (endCriteria.checkMaxIterations(gen, ecType))
                break;
            if (endCriteria.checkStationaryFunctionValue(fBestPrev, fBestGen, statState, ecType))
                break;
            fBestPrev = fBestGen;

            const Real maxD = *std::max_element(dDiag.begin(), dDiag.end());
            if (sigma * maxD < endCriteria.rootEpsilon() &&
                sigma * Norm2(pCov) < endCriteria.rootEpsilon()) {
                ecType = EndCriteria::StationaryPoint;
                break;
            }
            if (!(sigma > 0.0) || sigma > maxReal) { // sigma under/overflow
                ecType = EndCriteria::StationaryPoint;
                break;
            }

            // lazy refresh of the eigen-system B, D, C^{-1/2} from C
            if (gen % lazyGens == 0) {
                for (Size i = 0; i < n; ++i)
                    for (Size j = 0; j < i; ++j)
                        C[i][j] = C[j][i] = 0.5 * (C[i][j] + C[j][i]);
                SymmetricSchurDecomposition jacobi(C);
                const Array& eig = jacobi.eigenvalues();
                B = jacobi.eigenvectors();
                const Real maxEig = *std::max_element(eig.begin(), eig.end());
                const Real minEig = *std::min_element(eig.begin(), eig.end());
                // a blown-up condition number of C is a stagnation stop
                if (minEig <= 0.0 || maxEig / std::max(minEig, QL_EPSILON) > 1.0e14) {
                    ecType = EndCriteria::StationaryPoint;
                    break;
                }
                for (Size i = 0; i < n; ++i)
                    dDiag[i] = std::sqrt(eig[i]);
                Matrix BD(n, n); // invSqrtC = B * diag(1/D) * B^T
                for (Size i = 0; i < n; ++i)
                    for (Size j = 0; j < n; ++j)
                        BD[i][j] = B[i][j] / dDiag[j];
                invSqrtC = BD * transpose(B);
            }
        } while (true);

        P.setCurrentValue(bestX);
        P.setFunctionValue(bestF);
        return ecType;
    }

}
