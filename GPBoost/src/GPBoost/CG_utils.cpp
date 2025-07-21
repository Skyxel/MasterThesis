/*!
* This file is part of GPBoost a C++ library for combining
*	boosting with Gaussian process and mixed effects models
*
* Copyright (c) 2022 - 2024 Tim Gyger, Pascal Kuendig, and Fabio Sigrist. All rights reserved.
*
* Licensed under the Apache License Version 2.0. See LICENSE file in the project root for license information.
*/
#include <GPBoost/CG_utils.h>
#include <GPBoost/type_defs.h>
#include <GPBoost/re_comp.h>
#include <LightGBM/utils/log.h>

#include <chrono>
#include <thread> //temp

using LightGBM::Log;

namespace GPBoost {

	void CGVecchiaLaplaceVec(const vec_t& diag_W,
		const sp_mat_rm_t& B_rm,
		const sp_mat_rm_t& B_t_D_inv_rm,
		const vec_t& rhs,
		vec_t& u,
		bool& NA_or_Inf_found,
		int p,
		const int find_mode_it,
		const double delta_conv,
		const double THRESHOLD_ZERO_RHS_CG,
		const string_t cg_preconditioner_type,
		const sp_mat_rm_t& D_inv_plus_W_B_rm,
		const sp_mat_rm_t& L_SigmaI_plus_W_rm,
		const sp_mat_rm_t& P_H_sqrt_rm,
		const den_mat_t& L_k,
		const vec_t& d_w_plus_I,
		const chol_den_mat_t& chol_fact_I_k_plus_L_kt_D_W_plus_I_L_k_vecchia,
		bool run_in_parallel_do_not_report_non_convergence) {

		p = std::min(p, (int)B_rm.cols());

		vec_t r, r_old;
		vec_t z, z_old;
		vec_t h, v, B_invt_r, L_invt_r;
		vec_t P_H_sqrt_inv_r, P_L_inv_P_H_sqrt_inv_r, d_w_plus_I_P_H_sqrt_inv_r, Lkt_d_w_plus_I_P_H_sqrt_inv_r;
		double a, b, r_norm;
		
		//Avoid numerical instabilites when rhs is de facto 0
		if (rhs.cwiseAbs().sum() < THRESHOLD_ZERO_RHS_CG) {
			u.setZero();
			return;
		}

		//Cold-start in the first iteration of mode finding, otherwise always warm-start (=initalize with mode from previous iteration)
		if (find_mode_it == 0) {
			u.setZero();
			//r = rhs - A * u
			r = rhs; //since u is 0
		}
		else {
			//r = rhs - A * u
			r = rhs - ((B_t_D_inv_rm * (B_rm * u)) + diag_W.cwiseProduct(u));
		}

		if (cg_preconditioner_type == "vadu") {
			//z = P^(-1) r, where P^(-1) = B^(-1) (D^(-1) + W)^(-1) B^(-T)
			B_invt_r = B_rm.transpose().triangularView<Eigen::UpLoType::UnitUpper>().solve(r);
			z = D_inv_plus_W_B_rm.triangularView<Eigen::UpLoType::Lower>().solve(B_invt_r);
		}else if(cg_preconditioner_type == "hlfpc" ||
				 cg_preconditioner_type == "hlfpc_nystroem_last" ||
				 cg_preconditioner_type == "hlfpc_nystroem_random" ||
				 cg_preconditioner_type == "hlfpc_pivoted_cholesky" ||
				 cg_preconditioner_type == "hlfpc_lanczos" ||
				 cg_preconditioner_type == "hlfpc_rlra"){
			// z = P^(-1) r, where P^(-1) = P_H^(-T/2) P_L^(-1) P_H^(-1/2)
			// P_H^(-1/2) r
			P_H_sqrt_inv_r = P_H_sqrt_rm.triangularView<Eigen::UpLoType::Upper>().solve(r);

			// P_L^(-1) P_H^(-1/2) r
			d_w_plus_I_P_H_sqrt_inv_r = d_w_plus_I.cwiseProduct(P_H_sqrt_inv_r);
			Lkt_d_w_plus_I_P_H_sqrt_inv_r = L_k.transpose() * d_w_plus_I_P_H_sqrt_inv_r;
			P_L_inv_P_H_sqrt_inv_r = d_w_plus_I_P_H_sqrt_inv_r - d_w_plus_I.asDiagonal() * L_k * chol_fact_I_k_plus_L_kt_D_W_plus_I_L_k_vecchia.solve(Lkt_d_w_plus_I_P_H_sqrt_inv_r);

			// P_H^(-T/2) P_L^(-1) P_H^(-1/2) r
			z = P_H_sqrt_rm.transpose().triangularView<Eigen::UpLoType::Lower>().solve(P_L_inv_P_H_sqrt_inv_r);
		}
		else if (cg_preconditioner_type == "incomplete_cholesky" || 
			cg_preconditioner_type == "incomplete_cholesky_SN_A" ||
			cg_preconditioner_type == "incomplete_cholesky_SN_B" ||
			cg_preconditioner_type == "incomplete_cholesky_ABS" ||
			cg_preconditioner_type == "incomplete_cholesky_JM"||
			cg_preconditioner_type == "incomplete_cholesky_T"||
			cg_preconditioner_type == "incomplete_cholesky_TJM") {
			//z = P^(-1) r, where P^(-1) = L^(-1) L^(-T)
			L_invt_r = L_SigmaI_plus_W_rm.transpose().triangularView<Eigen::UpLoType::Upper>().solve(r);
			z = L_SigmaI_plus_W_rm.triangularView<Eigen::UpLoType::Lower>().solve(L_invt_r);
		}
		else {
			Log::REFatal("CGVecchiaLaplaceVec: Preconditioner type '%s' is not supported ", cg_preconditioner_type.c_str());
		}

		h = z;

		for (int j = 0; j < p; ++j) {
			//Parentheses are necessery for performance, otherwise EIGEN does the operation wrongly from left to right
			v = (B_t_D_inv_rm * (B_rm * h)) + diag_W.cwiseProduct(h);

			a = r.transpose() * z;
			a /= h.transpose() * v;

			u += a * h;
			r_old = r;
			r -= a * v;

			r_norm = r.norm();
			//Log::REInfo("r.norm(): %g | Iteration: %i", r_norm, j);
			if (std::isnan(r_norm) || std::isinf(r_norm)) {
				NA_or_Inf_found = true;
				return;
			}
			if (r_norm < delta_conv) {
				//Log::REInfo("Number CG iterations: %i", j + 1);
				return;
			}

			z_old = z;

			if (cg_preconditioner_type == "vadu") {
				//z = P^(-1) r 
				B_invt_r = B_rm.transpose().triangularView<Eigen::UpLoType::UnitUpper>().solve(r);
				z = D_inv_plus_W_B_rm.triangularView<Eigen::UpLoType::Lower>().solve(B_invt_r);
			}
			else if(cg_preconditioner_type == "hlfpc" ||
					cg_preconditioner_type == "hlfpc_nystroem_last" ||
					cg_preconditioner_type == "hlfpc_nystroem_random" ||
					cg_preconditioner_type == "hlfpc_pivoted_cholesky" ||
					cg_preconditioner_type == "hlfpc_lanczos" ||
					cg_preconditioner_type == "hlfpc_rlra"){
				// z = P^(-1) r, where P^(-1) = P_H^(-T/2) P_L^(-1) P_H^(-1/2)
				// P_H^(-1/2) r
				P_H_sqrt_inv_r = P_H_sqrt_rm.triangularView<Eigen::UpLoType::Upper>().solve(r);

				// P_L^(-1) P_H^(-1/2) r
				d_w_plus_I_P_H_sqrt_inv_r = d_w_plus_I.cwiseProduct(P_H_sqrt_inv_r);
				Lkt_d_w_plus_I_P_H_sqrt_inv_r = L_k.transpose() * d_w_plus_I_P_H_sqrt_inv_r;
				P_L_inv_P_H_sqrt_inv_r = d_w_plus_I_P_H_sqrt_inv_r - d_w_plus_I.asDiagonal() * L_k * chol_fact_I_k_plus_L_kt_D_W_plus_I_L_k_vecchia.solve(Lkt_d_w_plus_I_P_H_sqrt_inv_r);

				// P_H^(-T/2) P_L^(-1) P_H^(-1/2) r
				z = P_H_sqrt_rm.transpose().triangularView<Eigen::UpLoType::Lower>().solve(P_L_inv_P_H_sqrt_inv_r);
			}
			else if (cg_preconditioner_type == "incomplete_cholesky" || 
				cg_preconditioner_type == "incomplete_cholesky_SN_A" ||
				cg_preconditioner_type == "incomplete_cholesky_SN_B" ||
				cg_preconditioner_type == "incomplete_cholesky_ABS" ||
				cg_preconditioner_type == "incomplete_cholesky_JM" ||
				cg_preconditioner_type == "incomplete_cholesky_T" ||
				cg_preconditioner_type == "incomplete_cholesky_TJM") {
				//z = P^(-1) r, where P^(-1) = L^(-1) L^(-T)
				L_invt_r = L_SigmaI_plus_W_rm.transpose().triangularView<Eigen::UpLoType::Upper>().solve(r);
				z = L_SigmaI_plus_W_rm.triangularView<Eigen::UpLoType::Lower>().solve(L_invt_r);
			}
			else {
				Log::REFatal("CGVecchiaLaplaceVec: Preconditioner type '%s' is not supported ", cg_preconditioner_type.c_str());
			}

			b = r.transpose() * z;
			b /= r_old.transpose() * z_old;

			h = z + b * h;
		}
		if (!run_in_parallel_do_not_report_non_convergence) {
			Log::REDebug("Conjugate gradient algorithm has not converged after the maximal number of iterations (%i). "
				"This could happen if the initial learning rate is too large in a line search phase. Otherwise you might increase 'cg_max_num_it' ", p);
		}
	} // end CGVecchiaLaplaceVec

	void CGVecchiaLaplaceVecWinvplusSigma(const vec_t& diag_W,
		const sp_mat_rm_t& B_rm,
		const sp_mat_rm_t& D_inv_B_rm,
		const vec_t& rhs,
		vec_t& u,
		bool& NA_or_Inf_found,
		int p,
		const int find_mode_it,
		const double delta_conv,
		const double THRESHOLD_ZERO_RHS_CG,
		const chol_den_mat_t& chol_fact_I_k_plus_Sigma_L_kt_W_Sigma_L_k_vecchia,
		const den_mat_t& Sigma_L_k,
		bool run_in_parallel_do_not_report_non_convergence) {

		p = std::min(p, (int)B_rm.cols());

		CHECK(Sigma_L_k.rows() == B_rm.cols());
		CHECK(Sigma_L_k.rows() == diag_W.size());

		vec_t r, r_old;
		vec_t z, z_old;
		vec_t h, v, diag_W_inv, B_invt_u, B_invt_h, B_invt_rhs, Sigma_Lkt_W_r, Sigma_rhs, W_r;
		double a, b, r_norm;

		//Avoid numerical instabilites when rhs is de facto 0
		if (rhs.cwiseAbs().sum() < THRESHOLD_ZERO_RHS_CG) {
			u.setZero();
			return;
		}

		diag_W_inv = diag_W.cwiseInverse();

		//Sigma * rhs, where Sigma = B^(-1) D B^(-T)
		B_invt_rhs = B_rm.transpose().triangularView<Eigen::UpLoType::UnitUpper>().solve(rhs);
		Sigma_rhs = D_inv_B_rm.triangularView<Eigen::UpLoType::Lower>().solve(B_invt_rhs);

		//Cold-start in the first iteration of mode finding, otherwise always warm-start (=initalize with mode from previous iteration)
		if (find_mode_it == 0) {
			u.setZero();
			r = Sigma_rhs; //since u is 0
		}
		else {
			//r = Sigma * rhs - (W^(-1) + Sigma) * W * u // note: u contains the previous solution to (W + Sigma)^(-1)u = rhs, but here we solve (W^(-1) + Sigma) W u = Sigma * rhs, i.e. W * u is our initial vector
			r = Sigma_rhs - u;
			u = diag_W.cwiseProduct(u);
			vec_t B_invt_W_u = B_rm.transpose().triangularView<Eigen::UpLoType::UnitUpper>().solve(u);
			r = r - D_inv_B_rm.triangularView<Eigen::UpLoType::Lower>().solve(B_invt_W_u);
		}

		//z = P^(-1) r 
		//P^(-1) = (W^(-1) + Sigma_L_k Sigma_L_k^T)^(-1) = W - W Sigma_L_k (I_k + Sigma_L_k^T W Sigma_L_k)^(-1) Sigma_L_k^T W
		W_r = diag_W.asDiagonal() * r;
		Sigma_Lkt_W_r = Sigma_L_k.transpose() * W_r;
		//No case distinction for the brackets since Sigma_L_k is dense
		z = W_r - diag_W.asDiagonal() * (Sigma_L_k * chol_fact_I_k_plus_Sigma_L_kt_W_Sigma_L_k_vecchia.solve(Sigma_Lkt_W_r));

		h = z;
		
		for (int j = 0; j < p; ++j) {
			//(W^(-1) + Sigma) * h
			B_invt_h = B_rm.transpose().triangularView<Eigen::UpLoType::UnitUpper>().solve(h);
			v = D_inv_B_rm.triangularView<Eigen::UpLoType::Lower>().solve(B_invt_h) + diag_W_inv.cwiseProduct(h);

			a = r.transpose() * z;
			a /= h.transpose() * v;

			u += a * h;
			r_old = r;
			r -= a * v;

			r_norm = r.norm();
			if (std::isnan(r_norm) || std::isinf(r_norm)) {
				NA_or_Inf_found = true;
				return;
			}
			if (r_norm < delta_conv || (j + 1) == p) {
				//u = W^(-1) u
				u = diag_W_inv.cwiseProduct(u);
				if ((j + 1) == p) {
					if (!run_in_parallel_do_not_report_non_convergence) {
						Log::REDebug("Conjugate gradient algorithm has not converged after the maximal number of iterations (%i). "
							"This could happen if the initial learning rate is too large in a line search phase. Otherwise you might increase 'cg_max_num_it' ", p);
					}
				}
				//Log::REInfo("Number CG iterations: %i", j + 1);//for debugging
				return;
			}

			z_old = z;

			//z = P^(-1) r
			W_r = diag_W.asDiagonal() * r;
			Sigma_Lkt_W_r = Sigma_L_k.transpose() * W_r;
			z = W_r - diag_W.asDiagonal() * (Sigma_L_k * chol_fact_I_k_plus_Sigma_L_kt_W_Sigma_L_k_vecchia.solve(Sigma_Lkt_W_r));

			b = r.transpose() * z;
			b /= r_old.transpose() * z_old;

			h = z + b * h;
		}
	} // end CGVecchiaLaplaceVecSigmaplusWinv


	void CGVecchiaLaplaceVecWinvplusSigma_FITC_P(const vec_t& diag_W,
		const sp_mat_rm_t& B_rm,
		const sp_mat_rm_t& D_inv_B_rm,
		const vec_t& rhs,
		vec_t& u,
		bool& NA_or_Inf_found,
		int p,
		const int find_mode_it,
		const double delta_conv,
		const double THRESHOLD_ZERO_RHS_CG,
		const chol_den_mat_t& chol_fact_woodbury_preconditioner,
		const den_mat_t cross_cov,
		const vec_t& diagonal_approx_inv_preconditioner,
		bool run_in_parallel_do_not_report_non_convergence) {

		p = std::min(p, (int)B_rm.cols());

		CHECK((cross_cov).rows() == B_rm.cols());
		CHECK((cross_cov).rows() == diag_W.size());

		vec_t r, r_old;
		vec_t z, z_old;
		vec_t h, v, diag_W_inv, B_invt_u, B_invt_h, B_invt_rhs, Sigma_rhs, W_r;
		double a, b, r_norm;

		//Avoid numerical instabilites when rhs is de facto 0
		if (rhs.cwiseAbs().sum() < THRESHOLD_ZERO_RHS_CG) {
			u.setZero();
			return;
		}

		diag_W_inv = diag_W.cwiseInverse();

		//Sigma * rhs, where Sigma = B^(-1) D B^(-T)
		B_invt_rhs = B_rm.transpose().triangularView<Eigen::UpLoType::UnitUpper>().solve(rhs);
		Sigma_rhs = D_inv_B_rm.triangularView<Eigen::UpLoType::Lower>().solve(B_invt_rhs);

		//Cold-start in the first iteration of mode finding, otherwise always warm-start (=initalize with mode from previous iteration)
		if (find_mode_it == 0) {
			u.setZero();
			//r = Sigma * rhs - (W^(-1) + Sigma) * u
			r = Sigma_rhs; //since u is 0
		}
		else {
			//r = Sigma * rhs - (W^(-1) + Sigma) * W * u // note: u contains the previous solution to (W + Sigma)^(-1)u = rhs, but here we solve (W^(-1) + Sigma) W u = Sigma * rhs, i.e. W * u is our initial vector
			r = Sigma_rhs - u;
			u = diag_W.cwiseProduct(u);
			vec_t B_invt_W_u = B_rm.transpose().triangularView<Eigen::UpLoType::UnitUpper>().solve(u);
			r = r - D_inv_B_rm.triangularView<Eigen::UpLoType::Lower>().solve(B_invt_W_u);
		}

		//z = P^(-1) r 
		W_r = diagonal_approx_inv_preconditioner.asDiagonal() * r;
		//No case distinction for the brackets since Sigma_L_k is dense
		z = W_r - diagonal_approx_inv_preconditioner.asDiagonal() * ((cross_cov) * chol_fact_woodbury_preconditioner.solve((cross_cov).transpose() * W_r));

		h = z;
		
		for (int j = 0; j < p; ++j) {
			//(W^(-1) + Sigma) * h
			B_invt_h = B_rm.transpose().triangularView<Eigen::UpLoType::UnitUpper>().solve(h);
			v = D_inv_B_rm.triangularView<Eigen::UpLoType::Lower>().solve(B_invt_h) + diag_W_inv.cwiseProduct(h);

			a = r.transpose() * z;
			a /= h.transpose() * v;

			u += a * h;
			r_old = r;
			r -= a * v;

			r_norm = r.norm();
			if (std::isnan(r_norm) || std::isinf(r_norm)) {
				NA_or_Inf_found = true;
				return;
			}
			if (r_norm < delta_conv || (j + 1) == p) {
				//u = W^(-1) u
				u = diag_W_inv.cwiseProduct(u);
				if ((j + 1) == p) {
					if (!run_in_parallel_do_not_report_non_convergence) {
						Log::REDebug("Conjugate gradient algorithm has not converged after the maximal number of iterations (%i). "
							"This could happen if the initial learning rate is too large in a line search phase. Otherwise you might increase 'cg_max_num_it' ", p);
					}
				}
				//Log::REInfo("Number CG iterations: %i", j + 1);//for debugging
				return;
			}

			z_old = z;

			//z = P^(-1) r
			W_r = diagonal_approx_inv_preconditioner.asDiagonal() * r;
			//No case distinction for the brackets since Sigma_L_k is dense
			z = W_r - diagonal_approx_inv_preconditioner.asDiagonal() * ((cross_cov) * chol_fact_woodbury_preconditioner.solve((cross_cov).transpose() * W_r));



			b = r.transpose() * z;
			b /= r_old.transpose() * z_old;

			h = z + b * h;
		}
	} // end CGVecchiaLaplaceVecWinvplusSigma_FITC_P

	void CGTridiagVecchiaLaplace(const vec_t& diag_W,
		const sp_mat_rm_t& B_rm,
		const sp_mat_rm_t& B_t_D_inv_rm,
		const den_mat_t& rhs,
		std::vector<vec_t>& Tdiags,
		std::vector<vec_t>& Tsubdiags,
		den_mat_t& U,
		bool& NA_or_Inf_found,
		const data_size_t num_data,
		const int t,
		int p,
		const double delta_conv,
		const string_t cg_preconditioner_type,
		const sp_mat_rm_t& D_inv_plus_W_B_rm,
		const sp_mat_rm_t& L_SigmaI_plus_W_rm,
		const sp_mat_rm_t& P_H_sqrt_rm,
		const den_mat_t& L_k,
		const vec_t& d_w_plus_I,
		const chol_den_mat_t& chol_fact_I_k_plus_L_kt_D_W_plus_I_L_k_vecchia) {

		p = std::min(p, (int)num_data);

		den_mat_t R(num_data, t), R_old, P_sqrt_invt_R(num_data, t), Z(num_data, t), Z_old, H, V(num_data, t), L_kt_W_inv_R, B_k_W_inv_R, W_inv_R;
		den_mat_t P_H_sqrt_inv_r(num_data, t), P_L_inv_P_H_sqrt_inv_r(num_data, t), D_W_plus_I_P_H_sqrt_inv_r, Lkt_D_W_plus_I_P_H_sqrt_inv_r;
		vec_t v1(num_data), diag_SigmaI_plus_W_inv, diag_W_inv;
		vec_t a(t), a_old(t);
		vec_t b(t), b_old(t);
		bool early_stop_alg = false;
		double mean_R_norm;

		U.setZero();
		v1.setOnes();
		a.setOnes();
		b.setZero();

		//R = rhs - (W^(-1) + Sigma) * U
		R = rhs; //Since U is 0

		if (cg_preconditioner_type == "vadu") {
			//Z = P^(-1) R 		
			//P^(-1) = B^(-1) (D^(-1) + W)^(-1) B^(-T)
#pragma omp parallel for schedule(static)   
			for (int i = 0; i < t; ++i) {
				P_sqrt_invt_R.col(i) = B_rm.transpose().triangularView<Eigen::UpLoType::UnitUpper>().solve(R.col(i));
			}
#pragma omp parallel for schedule(static)   
			for (int i = 0; i < t; ++i) {
				Z.col(i) = D_inv_plus_W_B_rm.triangularView<Eigen::UpLoType::Lower>().solve(P_sqrt_invt_R.col(i));
			}
		}
		else if (cg_preconditioner_type == "hlfpc" ||
				 cg_preconditioner_type == "hlfpc_nystroem_last" ||
				 cg_preconditioner_type == "hlfpc_nystroem_random" ||
				 cg_preconditioner_type == "hlfpc_pivoted_cholesky" ||
				 cg_preconditioner_type == "hlfpc_lanczos" ||
				 cg_preconditioner_type == "hlfpc_rlra") {
			// Z = P^(-1) R, where P^(-1) = P_H^(-T/2) P_L^(-1) P_H^(-1/2)
 
			// P_H^(-1/2) r
#pragma omp parallel for schedule(static) 
			for (int i = 0; i < t; ++i) {
				P_H_sqrt_inv_r.col(i) = P_H_sqrt_rm.triangularView<Eigen::UpLoType::Upper>().solve(R.col(i));
			}

			// P_L^(-1) P_H^(-1/2) r
			D_W_plus_I_P_H_sqrt_inv_r = d_w_plus_I.asDiagonal() * P_H_sqrt_inv_r;
			Lkt_D_W_plus_I_P_H_sqrt_inv_r = L_k.transpose() * D_W_plus_I_P_H_sqrt_inv_r;
			P_L_inv_P_H_sqrt_inv_r = D_W_plus_I_P_H_sqrt_inv_r - d_w_plus_I.asDiagonal() * L_k * chol_fact_I_k_plus_L_kt_D_W_plus_I_L_k_vecchia.solve(Lkt_D_W_plus_I_P_H_sqrt_inv_r);

			// P_H^(-T/2) P_L^(-1) P_H^(-1/2) r  
#pragma omp parallel for schedule(static) 
			for (int i = 0; i < t; ++i) {
				Z.col(i) = P_H_sqrt_rm.transpose().triangularView<Eigen::UpLoType::Lower>().solve(P_L_inv_P_H_sqrt_inv_r.col(i));
			}

		}
		else if (cg_preconditioner_type == "incomplete_cholesky" ||
			cg_preconditioner_type == "incomplete_cholesky_SN_A" ||
			cg_preconditioner_type == "incomplete_cholesky_SN_B" ||
			cg_preconditioner_type == "incomplete_cholesky_ABS" ||
			cg_preconditioner_type == "incomplete_cholesky_JM" ||
			cg_preconditioner_type == "incomplete_cholesky_T" ||
			cg_preconditioner_type == "incomplete_cholesky_TJM") {
			//Z = P^(-1) R 		
			//P^(-1) = L^(-1) L^(-T)
#pragma omp parallel for schedule(static)   
			for (int i = 0; i < t; ++i) {
				P_sqrt_invt_R.col(i) = L_SigmaI_plus_W_rm.transpose().triangularView<Eigen::UpLoType::Upper>().solve(R.col(i));
			}
#pragma omp parallel for schedule(static)   
			for (int i = 0; i < t; ++i) {
				Z.col(i) = L_SigmaI_plus_W_rm.triangularView<Eigen::UpLoType::Lower>().solve(P_sqrt_invt_R.col(i));
			}
		}
		else {
			Log::REFatal("CGTridiagVecchiaLaplace: Preconditioner type '%s' is not supported ", cg_preconditioner_type.c_str());
		}

		H = Z;

		for (int j = 0; j < p; ++j) {
			//V = (Sigma^(-1) + W) H
#pragma omp parallel for schedule(static)   
			for (int i = 0; i < t; ++i) {
				V.col(i) = (B_t_D_inv_rm * (B_rm * H.col(i))) + diag_W.cwiseProduct(H.col(i));
			}

			a_old = a;
			a = (R.cwiseProduct(Z).transpose() * v1).array() * (H.cwiseProduct(V).transpose() * v1).array().inverse(); //cheap

			U += H * a.asDiagonal();
			R_old = R;
			R -= V * a.asDiagonal();

			mean_R_norm = R.colwise().norm().mean();

			if (std::isnan(mean_R_norm) || std::isinf(mean_R_norm)) {
				NA_or_Inf_found = true;
				return;
			}
			if (mean_R_norm < delta_conv) {
				early_stop_alg = true;
				//Log::REInfo("Number CG-Tridiag iterations: %i", j + 1);
			}

			Z_old = Z;

			//Z = P^(-1) R
			if (cg_preconditioner_type == "vadu") {
#pragma omp parallel for schedule(static)   
				for (int i = 0; i < t; ++i) {
					P_sqrt_invt_R.col(i) = B_rm.transpose().triangularView<Eigen::UpLoType::UnitUpper>().solve(R.col(i));
				}
#pragma omp parallel for schedule(static)   
				for (int i = 0; i < t; ++i) {
					Z.col(i) = D_inv_plus_W_B_rm.triangularView<Eigen::UpLoType::Lower>().solve(P_sqrt_invt_R.col(i));
				}
			}
			else if (cg_preconditioner_type == "hlfpc" ||
					 cg_preconditioner_type == "hlfpc_nystroem_last" ||
					 cg_preconditioner_type == "hlfpc_nystroem_random" ||
					 cg_preconditioner_type == "hlfpc_pivoted_cholesky" ||
					 cg_preconditioner_type == "hlfpc_lanczos" ||
					 cg_preconditioner_type == "hlfpc_rlra") {
				// Z = P^(-1) R, where P^(-1) = P_H^(-T/2) P_L^(-1) P_H^(-1/2)

				// P_H^(-1/2) r
#pragma omp parallel for schedule(static)   
				for (int i = 0; i < t; ++i) {
					P_H_sqrt_inv_r.col(i) = P_H_sqrt_rm.triangularView<Eigen::UpLoType::Upper>().solve(R.col(i));
				}

				// P_L^(-1) P_H^(-1/2) r
				D_W_plus_I_P_H_sqrt_inv_r = d_w_plus_I.asDiagonal() * P_H_sqrt_inv_r;
				Lkt_D_W_plus_I_P_H_sqrt_inv_r = L_k.transpose() * D_W_plus_I_P_H_sqrt_inv_r;
				P_L_inv_P_H_sqrt_inv_r = D_W_plus_I_P_H_sqrt_inv_r - d_w_plus_I.asDiagonal() * L_k * chol_fact_I_k_plus_L_kt_D_W_plus_I_L_k_vecchia.solve(Lkt_D_W_plus_I_P_H_sqrt_inv_r);


				// P_H^(-T/2) P_L^(-1) P_H^(-1/2) r  
#pragma omp parallel for schedule(static) 
				for (int i = 0; i < t; ++i) {
					Z.col(i) = P_H_sqrt_rm.transpose().triangularView<Eigen::UpLoType::Lower>().solve(P_L_inv_P_H_sqrt_inv_r.col(i));
				}

			}
			else if (cg_preconditioner_type == "incomplete_cholesky" || 
				cg_preconditioner_type == "incomplete_cholesky_SN_A" || 
				cg_preconditioner_type == "incomplete_cholesky_SN_B" ||
				cg_preconditioner_type == "incomplete_cholesky_ABS" ||
				cg_preconditioner_type == "incomplete_cholesky_JM" ||
				cg_preconditioner_type == "incomplete_cholesky_T" ||
				cg_preconditioner_type == "incomplete_cholesky_TJM") {
#pragma omp parallel for schedule(static)   
				for (int i = 0; i < t; ++i) {
					P_sqrt_invt_R.col(i) = L_SigmaI_plus_W_rm.transpose().triangularView<Eigen::UpLoType::Upper>().solve(R.col(i));
				}
#pragma omp parallel for schedule(static)   
				for (int i = 0; i < t; ++i) {
					Z.col(i) = L_SigmaI_plus_W_rm.triangularView<Eigen::UpLoType::Lower>().solve(P_sqrt_invt_R.col(i));
				}
			}
			else {
				Log::REFatal("CGTridiagVecchiaLaplace: Preconditioner type '%s' is not supported ", cg_preconditioner_type.c_str());
			}

			b_old = b;
			b = (R.cwiseProduct(Z).transpose() * v1).array() * (R_old.cwiseProduct(Z_old).transpose() * v1).array().inverse();

			H = Z + H * b.asDiagonal();

#pragma omp parallel for schedule(static)
			for (int i = 0; i < t; ++i) {
				Tdiags[i][j] = 1 / a(i) + b_old(i) / a_old(i);
				if (j > 0) {
					Tsubdiags[i][j - 1] = sqrt(b_old(i)) / a_old(i);
				}
			}

			if (early_stop_alg) {
				for (int i = 0; i < t; ++i) {
					Tdiags[i].conservativeResize(j + 1, 1);
					Tsubdiags[i].conservativeResize(j, 1);
				}
				return;
			}
		}
		Log::REDebug("Conjugate gradient algorithm has not converged after the maximal number of iterations (%i). "
			"This could happen if the initial learning rate is too large. Otherwise you might increase 'cg_max_num_it_tridiag' ", p);
	} // end CGTridiagVecchiaLaplace

	void CGTridiagVecchiaLaplaceWinvplusSigma(const vec_t& diag_W,
		const sp_mat_rm_t& B_rm,
		const sp_mat_rm_t& D_inv_B_rm,
		const den_mat_t& rhs,
		std::vector<vec_t>& Tdiags,
		std::vector<vec_t>& Tsubdiags,
		den_mat_t& U,
		bool& NA_or_Inf_found,
		const data_size_t num_data,
		const int t,
		int p,
		const double delta_conv,
		const chol_den_mat_t& chol_fact_I_k_plus_Sigma_L_kt_W_Sigma_L_k_vecchia,
		const den_mat_t& Sigma_L_k) {

		p = std::min(p, (int)num_data);

		den_mat_t B_invt_U(num_data, t), Sigma_Lkt_W_R, B_invt_H(num_data, t), W_R;
		den_mat_t R(num_data, t), R_old, Z, Z_old, H, V(num_data, t);
		vec_t v1(num_data), diag_W_inv;
		vec_t a(t), a_old(t);
		vec_t b(t), b_old(t);
		bool early_stop_alg = false;
		double mean_R_norm;

		diag_W_inv = diag_W.cwiseInverse();
		U.setZero();
		v1.setOnes();
		a.setOnes();
		b.setZero();

		//R = rhs - (W^(-1) + Sigma) * U
		R = rhs; //Since U is 0

		//Z = P^(-1) R 
		//P^(-1) = (W^(-1) + Sigma_L_k Sigma_L_k^T)^(-1) = W - W Sigma_L_k (I_k + Sigma_L_k^T W Sigma_L_k)^(-1) Sigma_L_k^T W
		W_R = diag_W.asDiagonal() * R;
		Sigma_Lkt_W_R = Sigma_L_k.transpose() * W_R;
		if (Sigma_L_k.cols() < t) {
			Z = W_R - (diag_W.asDiagonal() * Sigma_L_k) * chol_fact_I_k_plus_Sigma_L_kt_W_Sigma_L_k_vecchia.solve(Sigma_Lkt_W_R);
		}
		else {
			Z = W_R - diag_W.asDiagonal() * (Sigma_L_k * chol_fact_I_k_plus_Sigma_L_kt_W_Sigma_L_k_vecchia.solve(Sigma_Lkt_W_R));
		}

		H = Z;

		for (int j = 0; j < p; ++j) {
			//V = (W^(-1) + Sigma) * H - expensive part of the loop
#pragma omp parallel for schedule(static)   
			for (int i = 0; i < t; ++i) {
				B_invt_H.col(i) = B_rm.transpose().triangularView<Eigen::UpLoType::UnitUpper>().solve(H.col(i));
			}
#pragma omp parallel for schedule(static)   
			for (int i = 0; i < t; ++i) {
				V.col(i) = D_inv_B_rm.triangularView<Eigen::UpLoType::Lower>().solve(B_invt_H.col(i));
			}
			V += diag_W_inv.replicate(1, t).cwiseProduct(H);

			a_old = a;
			a = (R.cwiseProduct(Z).transpose() * v1).array() * (H.cwiseProduct(V).transpose() * v1).array().inverse();

			U += H * a.asDiagonal();
			R_old = R;
			R -= V * a.asDiagonal();

			mean_R_norm = R.colwise().norm().mean();

			if (std::isnan(mean_R_norm) || std::isinf(mean_R_norm)) {
				NA_or_Inf_found = true;
				return;
			}
			if (mean_R_norm < delta_conv) {
				early_stop_alg = true;
				//Log::REInfo("Number CG-Tridiag iterations: %i", j + 1);
			}

			Z_old = Z;

			//Z = P^(-1) R
			W_R = diag_W.asDiagonal() * R;
			Sigma_Lkt_W_R = Sigma_L_k.transpose() * W_R;
			if (Sigma_L_k.cols() < t) {
				Z = W_R - (diag_W.asDiagonal() * Sigma_L_k) * chol_fact_I_k_plus_Sigma_L_kt_W_Sigma_L_k_vecchia.solve(Sigma_Lkt_W_R);
			}
			else {
				Z = W_R - diag_W.asDiagonal() * (Sigma_L_k * chol_fact_I_k_plus_Sigma_L_kt_W_Sigma_L_k_vecchia.solve(Sigma_Lkt_W_R));
			}
			b_old = b;
			b = (R.cwiseProduct(Z).transpose() * v1).array() * (R_old.cwiseProduct(Z_old).transpose() * v1).array().inverse();

			H = Z + H * b.asDiagonal();

#pragma omp parallel for schedule(static)
			for (int i = 0; i < t; ++i) {
				Tdiags[i][j] = 1 / a(i) + b_old(i) / a_old(i);
				if (j > 0) {
					Tsubdiags[i][j - 1] = sqrt(b_old(i)) / a_old(i);
				}
			}

			if (early_stop_alg) {
				for (int i = 0; i < t; ++i) {
					Tdiags[i].conservativeResize(j + 1, 1);
					Tsubdiags[i].conservativeResize(j, 1);
				}
				return;
			}
		}
		Log::REDebug("Conjugate gradient algorithm has not converged after the maximal number of iterations (%i). "
			"This could happen if the initial learning rate is too large. Otherwise you might increase 'cg_max_num_it_tridiag' ", p);
	} // end CGTridiagVecchiaLaplaceSigmaplusWinv


	void CGTridiagVecchiaLaplaceWinvplusSigma_FITC_P(const vec_t& diag_W,
		const sp_mat_rm_t& B_rm,
		const sp_mat_rm_t& D_inv_B_rm,
		const den_mat_t& rhs,
		std::vector<vec_t>& Tdiags,
		std::vector<vec_t>& Tsubdiags,
		den_mat_t& U,
		bool& NA_or_Inf_found,
		const data_size_t num_data,
		const int t,
		int p,
		const double delta_conv,
		const chol_den_mat_t& chol_fact_woodbury_preconditioner,
		const den_mat_t* cross_cov,
		const vec_t& diagonal_approx_inv_preconditioner) {

		p = std::min(p, (int)num_data);

		den_mat_t B_invt_U(num_data, t), Sigma_Lkt_W_R, B_invt_H(num_data, t), W_R;
		den_mat_t R(num_data, t), R_old, Z, Z_old, H, V(num_data, t);
		vec_t v1(num_data), diag_W_inv;
		vec_t a(t), a_old(t);
		vec_t b(t), b_old(t);
		bool early_stop_alg = false;
		double mean_R_norm;

		diag_W_inv = diag_W.cwiseInverse();
		U.setZero();
		v1.setOnes();
		a.setOnes();
		b.setZero();

		//R = rhs - (W^(-1) + Sigma) * U
		R = rhs; //Since U is 0

		//Z = P^(-1) R 
		W_R = diagonal_approx_inv_preconditioner.asDiagonal() * R;
		//No case distinction for the brackets since Sigma_L_k is dense
		Z = W_R - diagonal_approx_inv_preconditioner.asDiagonal() * ((*cross_cov) * chol_fact_woodbury_preconditioner.solve((*cross_cov).transpose() * W_R));

		H = Z;

		for (int j = 0; j < p; ++j) {
			//V = (W^(-1) + Sigma) * H - expensive part of the loop
#pragma omp parallel for schedule(static)   
			for (int i = 0; i < t; ++i) {
				B_invt_H.col(i) = B_rm.transpose().triangularView<Eigen::UpLoType::UnitUpper>().solve(H.col(i));
			}
#pragma omp parallel for schedule(static)   
			for (int i = 0; i < t; ++i) {
				V.col(i) = D_inv_B_rm.triangularView<Eigen::UpLoType::Lower>().solve(B_invt_H.col(i));
			}
			V += diag_W_inv.replicate(1, t).cwiseProduct(H);

			a_old = a;
			a = (R.cwiseProduct(Z).transpose() * v1).array() * (H.cwiseProduct(V).transpose() * v1).array().inverse();

			U += H * a.asDiagonal();
			R_old = R;
			R -= V * a.asDiagonal();

			mean_R_norm = R.colwise().norm().mean();

			if (std::isnan(mean_R_norm) || std::isinf(mean_R_norm)) {
				NA_or_Inf_found = true;
				return;
			}
			if (mean_R_norm < delta_conv) {
				early_stop_alg = true;
				//Log::REInfo("Number CG-Tridiag iterations: %i", j + 1);
			}

			Z_old = Z;

			//Z = P^(-1) R 
			W_R = diagonal_approx_inv_preconditioner.asDiagonal() * R;
			//No case distinction for the brackets since Sigma_L_k is dense
			Z = W_R - diagonal_approx_inv_preconditioner.asDiagonal() * ((*cross_cov) * chol_fact_woodbury_preconditioner.solve((*cross_cov).transpose() * W_R));

			b_old = b;
			b = (R.cwiseProduct(Z).transpose() * v1).array() * (R_old.cwiseProduct(Z_old).transpose() * v1).array().inverse();

			H = Z + H * b.asDiagonal();

#pragma omp parallel for schedule(static)
			for (int i = 0; i < t; ++i) {
				Tdiags[i][j] = 1 / a(i) + b_old(i) / a_old(i);
				if (j > 0) {
					Tsubdiags[i][j - 1] = sqrt(b_old(i)) / a_old(i);
				}
			}

			if (early_stop_alg) {
				for (int i = 0; i < t; ++i) {
					Tdiags[i].conservativeResize(j + 1, 1);
					Tsubdiags[i].conservativeResize(j, 1);
				}
				return;
			}
		}
		Log::REDebug("Conjugate gradient algorithm has not converged after the maximal number of iterations (%i). "
			"This could happen if the initial learning rate is too large. Otherwise you might increase 'cg_max_num_it_tridiag' ", p);
	} // end CGTridiagVecchiaLaplaceWinvplusSigma_FITC_P

	void simProbeVect(RNG_t& generator, den_mat_t& Z, const bool rademacher) {

		double u;

		if (rademacher) {
			std::uniform_real_distribution<double> udist(0.0, 1.0);

			for (int i = 0; i < Z.rows(); ++i) {
				for (int j = 0; j < Z.cols(); j++) {
					u = udist(generator);
					if (u > 0.5) {
						Z(i, j) = 1.;
					}
					else {
						Z(i, j) = -1.;
					}
				}
			}
		}
		else {
			std::normal_distribution<double> ndist(0.0, 1.0);

			for (int i = 0; i < Z.rows(); ++i) {
				for (int j = 0; j < Z.cols(); j++) {
					Z(i, j) = ndist(generator);
				}
			}
		}
	} // end simProbeVect

	void GenRandVecNormal(RNG_t& generator,
		den_mat_t& R) {
		std::normal_distribution<double> ndist(0.0, 1.0);
		//Do not parallelize! - Despite seed: no longer deterministic
		for (int i = 0; i < R.rows(); ++i) {
			for (int j = 0; j < R.cols(); j++) {
				R(i, j) = ndist(generator);
			}
		}
	}

	void GenRandVecRademacher(RNG_t& generator,
		den_mat_t& R) {
		double u;
		std::uniform_real_distribution<double> udist(0.0, 1.0);
		//Do not parallelize! - Despite seed: no longer deterministic
		for (int i = 0; i < R.rows(); ++i) {
			for (int j = 0; j < R.cols(); j++) {
				u = udist(generator);
				if (u > 0.5) {
					R(i, j) = 1.;
				}
				else {
					R(i, j) = -1.;
				}
			}
		}
	}

	void LogDetStochTridiag(const std::vector<vec_t>& Tdiags,
		const  std::vector<vec_t>& Tsubdiags,
		double& ldet,
		const data_size_t num_data,
		const int t) {

		Eigen::SelfAdjointEigenSolver<den_mat_t> es;
		ldet = 0;
		vec_t e1_logLambda_e1;

		for (int i = 0; i < t; ++i) {
			e1_logLambda_e1.setZero();
			es.computeFromTridiagonal(Tdiags[i], Tsubdiags[i]);
			e1_logLambda_e1 = es.eigenvectors().row(0).transpose().array() * es.eigenvalues().array().log() * es.eigenvectors().row(0).transpose().array();
			ldet += e1_logLambda_e1.sum();
		}
		ldet = ldet * num_data / t;
	} // end LogDetStochTridiag

	void CalcOptimalC(const vec_t& zt_AI_A_deriv_PI_z,
		const vec_t& zt_BI_B_deriv_PI_z,
		const double& tr_AI_A_deriv,
		const double& tr_BI_B_deriv, 
		double& c_opt) {

		vec_t centered_zt_AI_A_deriv_PI_z = zt_AI_A_deriv_PI_z.array() - tr_AI_A_deriv;
		vec_t centered_zt_BI_B_deriv_PI_z = zt_BI_B_deriv_PI_z.array() - tr_BI_B_deriv;
		c_opt = (centered_zt_AI_A_deriv_PI_z.cwiseProduct(centered_zt_BI_B_deriv_PI_z)).mean();
		c_opt /= (centered_zt_BI_B_deriv_PI_z.cwiseProduct(centered_zt_BI_B_deriv_PI_z)).mean();
	} // end CalcOptimalC

	void CalcOptimalCVectorized(const den_mat_t& Z_AI_A_deriv_PI_Z,
		const den_mat_t& Z_BI_B_deriv_PI_Z, 
		const vec_t& tr_AI_A_deriv,
		const vec_t& tr_BI_B_deriv,
		vec_t& c_opt) {

		den_mat_t centered_Z_AI_A_deriv_PI_Z = Z_AI_A_deriv_PI_Z.colwise() - tr_AI_A_deriv;
		den_mat_t centered_Z_BI_B_deriv_PI_Z = Z_BI_B_deriv_PI_Z.colwise() - tr_BI_B_deriv;
		vec_t c_cov = (centered_Z_AI_A_deriv_PI_Z.cwiseProduct(centered_Z_BI_B_deriv_PI_Z)).rowwise().mean();
		vec_t c_var = (centered_Z_BI_B_deriv_PI_Z.cwiseProduct(centered_Z_BI_B_deriv_PI_Z)).rowwise().mean();
		c_opt = c_cov.array() / c_var.array();
#pragma omp parallel for schedule(static)   
		for (int i = 0; i < c_opt.size(); ++i) {
			if (c_var.coeffRef(i) == 0) {
				c_opt[i] = 1;
			}
		}
	} // end CalcOptimalCVectorized

	void ReverseIncompleteCholeskyFactorization(sp_mat_t& A,
		const sp_mat_t& B,
		sp_mat_rm_t& L_rm,
		const string_t& preconditioner_type) {
		
		if(preconditioner_type == "incomplete_cholesky"){ // apply absolute value under square root and use sparsity pattern of B
			Log::REInfo("Using incomplete_cholesky");
		
			//Defining sparsity pattern 
			sp_mat_t L = A;
			//sp_mat_t L = B; //alternative version (less stable)
			
			L *= 0.0;

			for (int i = ((int)L.outerSize() - 1); i > -1; --i) {
				for (Eigen::SparseMatrix<double>::ReverseInnerIterator it(L, i); it; --it) {
					int j = (int)it.row();
					int ii = (int)it.col();
					double s = (L.col(j)).dot(L.col(ii));
					if (ii == j) {
						it.valueRef() = std::sqrt(A.coeffRef(ii, ii) + 1e-10 - s);
					}
					else if (ii < j) {
						it.valueRef() = (A.coeffRef(ii, j) - s) / L.coeffRef(j, j);
					}
					if (std::isnan(it.value()) || std::isinf(it.value())) {
						Log::REFatal("nan or inf occured in ReverseIncompleteCholeskyFactorization()");
					}
				}
			}

			L_rm = sp_mat_rm_t(L); //Convert to row-major

		}
		else if (preconditioner_type == "incomplete_cholesky_SN_A"){ // SUBSTITUTE NEGATIVE SQUARE ROOT AND INFINITE VALUES WITH SPARSITY PATTERN OF B
			Log::REInfo("Using incomplete_cholesky_SN_A");
		
			//Defining sparsity pattern 
			sp_mat_t L = A;
	
			L *= 0.0;

			double neg_sqrt_substitute = 1; 	// Substitute value for NaNs resulting from negative square roots

			for (int i = ((int)L.outerSize() - 1); i > -1; --i) { // Iterate on the columns
				for (Eigen::SparseMatrix<double>::ReverseInnerIterator it(L, i); it; --it) { // Iterate on the non null elements of the column i
					int j = (int)it.row();
					double aji = A.coeffRef(j,i);
					aji -= (L.col(j)).dot(L.col(i));
					if (i == j) {
						if(aji <= 0){
							Log::REInfo("column i = %d, row j = %d, value = %g", i, j, aji);
							L.coeffRef(i, i) = neg_sqrt_substitute;
						}else{
							L.coeffRef(i, i) = std::sqrt(aji);
						}
					}
					else if (i < j) {
						it.valueRef() = aji / L.coeffRef(j, j);
						if (std::isinf(it.value())) {
							Log::REInfo("column i = %d, row j = %d, value = %g", i, j, L.coeffRef(j, i));
							Log::REFatal("nan or inf occured in ReverseIncompleteCholeskyFactorization()");
						}
					}
				}
			}

			L_rm = sp_mat_rm_t(L); //Convert to row-major

		}
		else if (preconditioner_type == "incomplete_cholesky_SN_B"){ // SUBSTITUTE NEGATIVE SQUARE ROOT AND INFINITE VALUES WITH SPARSITY PATTERN OF B
			Log::REInfo("Using incomplete_cholesky_SN_B");
		
			//Defining sparsity pattern 
			sp_mat_t L = B;
	
			L *= 0.0;

			double neg_sqrt_substitute = 1; 	// Substitute value for NaNs resulting from negative square roots

			for (int i = ((int)L.outerSize() - 1); i > -1; --i) { // Iterate on the columns
				for (Eigen::SparseMatrix<double>::ReverseInnerIterator it(L, i); it; --it) { // Iterate on the non null elements of the column i
					int j = (int)it.row();
					double aji = A.coeffRef(j,i);
					aji -= (L.col(j)).dot(L.col(i));
					if (i == j) {
						L.coeffRef(i, i) = (aji <= 0) ? neg_sqrt_substitute : std::sqrt(aji);
					}
					else if (i < j) {
						it.valueRef() = aji / L.coeffRef(j, j);
						if (std::isinf(it.value())) {
							Log::REInfo("column i = %d, row j = %d, value = %g", i, j, L.coeffRef(j, i));
							Log::REFatal("nan or inf occured in ReverseIncompleteCholeskyFactorization()");
						}
					}
				}
			}

			L_rm = sp_mat_rm_t(L); //Convert to row-major

		}
		else if (preconditioner_type == "incomplete_cholesky_ABS"){ // SUBSTITUTE NEGATIVE SQUARE ROOT AND INFINITE VALUES WITH SPARSITY PATTERN OF B
			Log::REInfo("Using incomplete_cholesky_ABS");
		
			//Defining sparsity pattern 
			sp_mat_t L = B;
	
			L *= 0.0;

			for (int i = ((int)L.outerSize() - 1); i > -1; --i) { // Iterate on the columns
				for (Eigen::SparseMatrix<double>::ReverseInnerIterator it(L, i); it; --it) { // Iterate on the non null elements of the column i
					int j = (int)it.row();
					double aji = A.coeffRef(j,i);
					aji -= (L.col(j)).dot(L.col(i));
					if (i == j) {
						it.valueRef() = std::sqrt(std::abs(aji));
					}
					else if (i < j) {
						it.valueRef() = aji / L.coeffRef(j, j);
					}
				}
			}

			L_rm = sp_mat_rm_t(L); //Convert to row-major

		}
		else if(preconditioner_type == "incomplete_cholesky_T"){ // Tismenetsky scheme WITH SPARSITY PATTERN OF B
			Log::REInfo("Using incomplete_cholesky_T");
		
			//Defining sparsity pattern 
			sp_mat_t R = A;
			sp_mat_t L = B;
			
			L *= 0;
			R *= 0;

			double neg_sqrt_substitute = 1; 	// Substitute value for NaNs resulting from negative square roots

			// boolean matrix for L's pattern
			Eigen::SparseMatrix<bool> L_pattern(L.rows(), L.cols());

			for (int i = 0; i < L.outerSize(); ++i) {
				for (Eigen::SparseMatrix<double>::InnerIterator it(L, i); it; ++it) {
					int j = (int)it.row();
					if(i < j){
						L_pattern.coeffRef(j, i) = true;
					}
				}
			}

			// Tismenetsky scheme
			for (int i = ((int)L.outerSize() - 1); i > -1; --i) { // Iterate on the columns
				for (Eigen::SparseMatrix<double>::ReverseInnerIterator it(A, i); it; --it) { // Iterate on the non null elements of the column i
					int j = (int)it.row();
					if (i <= j){
						double aji = it.value();
						double s = (L.col(j)).dot(L.col(i)) + (R.col(j)).dot(L.col(i)) + (L.col(j)).dot(R.col(i));
						aji -= s;
						if(i == j){
							L.coeffRef(i, i) = (aji <= 0) ? neg_sqrt_substitute : std::sqrt(aji);
						}
						else if(L_pattern.coeff(j, i)){
							L.coeffRef(j, i) = aji / L.coeffRef(j, j);
						} else{
							R.coeffRef(j, i) = aji / L.coeffRef(j, j);
						}
					}
				}
			}
			
			L_rm = sp_mat_rm_t(L); //Convert to row-major
		}
		else if(preconditioner_type == "incomplete_cholesky_JM"){ // Jennings–Malik MODIFICATIONS WITH SPARSITY PATTERN OF B
			Log::REInfo("Using incomplete_cholesky_JM");
		
			//Defining sparsity pattern 
			sp_mat_t L = B;
			sp_mat_t A_modified = A;

			L *= 0;

			double neg_sqrt_substitute = 1; 	// Substitute value for NaNs resulting from negative square roots
			double gamma = 1.0;

			// boolean matrix for L's pattern
			Eigen::SparseMatrix<bool> L_pattern(L.rows(), L.cols());

			for (int i = 0; i < L.outerSize(); ++i) {
				for (Eigen::SparseMatrix<double>::InnerIterator it(L, i); it; ++it) {
					int j = (int)it.row();
					if(i < j){
						L_pattern.coeffRef(j, i) = true;
					}
				}
			}

			// Jennis-Malik modification
			for (int i = ((int)L.outerSize() - 1); i > -1; --i) {
				for (Eigen::SparseMatrix<double>::ReverseInnerIterator it(A_modified, i); it; --it) {
					int j = (int)it.row();
					if (i <= j){
						double aji = it.value();
						double s = (L.col(j)).dot(L.col(i));
						aji -= s;
						if(i == j){
							L.coeffRef(i, i) = (aji <= 0) ? neg_sqrt_substitute : std::sqrt(aji);
						} else if(L_pattern.coeff(j, i)){
							L.coeffRef(j, i) = aji / L.coeffRef(j, j);
						} else{
							A_modified.coeffRef(i,i) += (1 / gamma) * std::abs(aji);
						}
					}
				}
			}
			
			L_rm = sp_mat_rm_t(L); //Convert to row-major
		}
		else if(preconditioner_type == "incomplete_cholesky_TJM"){// Tismenetsky scheme with Jennings–Malik modifications WITH SPARSITY PATTERN OF B
			Log::REInfo("Using incomplete_cholesky_TJM");
		
			//Defining sparsity pattern 
			sp_mat_t R = A;
			sp_mat_t L = B;
			sp_mat_t A_modified = A;

			L *= 0;
			R *= 0;

			double neg_sqrt_substitute = 1;
			double gamma = 1.0;
			double droptol = 1e-3;

			// boolean matrix for L's pattern
			Eigen::SparseMatrix<bool> L_pattern(L.rows(), L.cols());

			for (int i = 0; i < L.outerSize(); ++i) {
				for (Eigen::SparseMatrix<double>::InnerIterator it(L, i); it; ++it) {
					int j = (int)it.row();
					if(i < j){
						L_pattern.coeffRef(j, i) = true;
					}
				}
			}

			// Tismenetsky scheme + Jennis-Malik modification
			for (int i = ((int)L.outerSize() - 1); i > -1; --i) { // Iterate on the columns
				for (Eigen::SparseMatrix<double>::ReverseInnerIterator it(A_modified, i); it; --it) { // Iterate on the non null elements of the column i
					int j = (int)it.row();
					if (i <= j){
						double aji = it.value();
						double s = (L.col(j)).dot(L.col(i)) + (R.col(j)).dot(L.col(i)) + (L.col(j)).dot(R.col(i));
						aji -= s;
						if(i == j){
							L.coeffRef(i, i) = (aji <= 0) ? neg_sqrt_substitute : std::sqrt(aji);
						}
						else if(L_pattern.coeff(j, i)){
							L.coeffRef(j, i) = aji / L.coeffRef(j, j);
							
						} else if(std::abs(aji) > droptol){
							R.coeffRef(j, i) = aji / L.coeffRef(j, j);
						} else{
							A_modified.coeffRef(i,i) += (1 / gamma) * std::abs(aji);
						}
					}
				}
			}
			
			L.prune(0.0);
			L_rm = sp_mat_rm_t(L); //Convert to row-major
		}
	} // end ReverseIncompleteCholeskyFactorization

	void PivotedCholsekyFactorizationForHLFPC(
		den_mat_t& L_k,
		const den_mat_t B_rm,
		const vec_t w_sqrt,
		const vec_t w_plus_d_inv_sqrt_inv,
		int max_it,
		data_size_t num_data,
		const double err_tol) {

		int m = 0;
		int i, pi_m_old;
		double err, L_jm;
		vec_t diag(num_data), L_row_m, e_i, L_h, L_i, L_j, v_1, v_2;
		vec_int_t pi(num_data);
		max_it = std::min(max_it, num_data);
		L_k.resize(num_data, max_it);
		L_k.setZero();

		for (int h = 0; h < num_data; ++h) {
			pi(h) = h;

			// Create the i-th unit vector
			e_i = vec_t::Zero(num_data);
			e_i(h) = 1.0;

			// find the i-th column of L_k
			v_1 = w_sqrt.cwiseProduct(e_i);
			v_2 = B_rm.transpose().triangularView<Eigen::UpLoType::UnitUpper>().solve(v_1);
			L_h = w_plus_d_inv_sqrt_inv.cwiseProduct(v_2);
			//The diagonal of the covariance matrix equals the marginal variance and is the same for all entries (i,i). 
			diag(h) = L_h.dot(L_h);
		}
		err = diag.lpNorm<1>();

		while (m == 0 || (m < max_it && err > err_tol)) {

			diag(pi.tail(num_data - m)).maxCoeff(&i);
			i += m;

			pi_m_old = pi(m);
			pi(m) = pi(i);
			pi(i) = pi_m_old;

			//L[(m+1):n,m]
			if ((m + 1) < num_data) {

				if (m > 0) {
					L_row_m = L_k.row(pi(m)).transpose();
				}

				for (int j = m + 1; j < num_data; ++j) {
					// Create the i-th unit vector
					e_i = vec_t::Zero(num_data);
					e_i(pi(m)) = 1.0;

					// find the i-th column of L_k
					v_1 = w_sqrt.cwiseProduct(e_i);
					v_2 = B_rm.transpose().triangularView<Eigen::UpLoType::UnitUpper>().solve(v_1);
					L_i = w_plus_d_inv_sqrt_inv.cwiseProduct(v_2);

					// Create the j-th unit vector
					e_i = vec_t::Zero(num_data);
					e_i(pi(j)) = 1.0;

					// find the j-th column of L_k
					v_1 = w_sqrt.cwiseProduct(e_i);
					v_2 = B_rm.transpose().triangularView<Eigen::UpLoType::UnitUpper>().solve(v_1);
					L_j = w_plus_d_inv_sqrt_inv.cwiseProduct(v_2);

					//The diagonal of the covariance matrix equals the marginal variance and is the same for all entries (i,i). 
					L_jm = L_j.dot(L_i);

					if (m > 0) { //need previous columns
						L_jm -= L_k.row(pi(j)).dot(L_row_m);
					}

					if (!(fabs(L_jm) < 1e-12)) {
						L_jm /= sqrt(diag(pi(m)));
						L_k(pi(j), m) = L_jm;
					}

					diag(pi(j)) -= L_jm * L_jm;
				}
				err = diag(pi.tail(num_data - (m + 1))).lpNorm<1>();
			}

			//L[m,m] - Update post L[(m+1):n,m] to be able to multiply with L[m,] beforehand
			L_k(pi(m), m) = sqrt(diag(pi(m)));
			m = m + 1;
		}
	}//end PivotedCholsekyFactorizationForHLFPC

	void LanczosTridiagVecchiaLaplaceNoPreconditioner(
		RNG_t& generator,
		const sp_mat_rm_t& B_rm,
		const vec_t& w,
		const vec_t& w_plus_d_inv_sqrt_inv,
		const data_size_t num_data,
		vec_t& Tdiag_k,
		vec_t& Tsubdiag_k,
		den_mat_t& Q_k,
		int& max_it,
		const double tol) {

		bool could_reorthogonalize;
		int final_rank = 1;
		double alpha_curr, beta_curr, beta_prev;
		vec_t q_curr, q_prev, inner_products, b_init;

		max_it = std::min(max_it, num_data);

		// Initialize b_init as b ~ N(0, I_n)
		b_init.setZero(num_data);
		std::normal_distribution<double> ndist(0.0, 1.0);
		//Do not parallelize! - Despite seed: no longer deterministic
		for (int i = 0; i < num_data; ++i) {
			b_init[i] = ndist(generator);
		}

		//Inital vector of Q_k: q_0
		Q_k.resize(num_data, max_it);
		vec_t q_0 = b_init / b_init.norm();
		Q_k.col(0) = q_0;

		//Initial alpha value: alpha_0
		//(W + D^(-1))^-0.5 * B-T * W * B^-1 * (W + D^(-1))^-0.5 q_0
		vec_t v_1 = w_plus_d_inv_sqrt_inv.cwiseProduct(q_0);
		vec_t v_2 = B_rm.triangularView<Eigen::UpLoType::UnitLower>().solve(v_1);
		vec_t v_3 = w.cwiseProduct(v_2);
		vec_t v_4 = B_rm.transpose().triangularView<Eigen::UpLoType::UnitUpper>().solve(v_3);
		vec_t r = w_plus_d_inv_sqrt_inv.cwiseProduct(v_4);
		double alpha_0 = q_0.dot(r);

		//Initial beta value: beta_0
		r -= alpha_0 * q_0;
		double beta_0 = r.norm();

		//Store alpha_0 and beta_0 into T_k
		Tdiag_k(0) = alpha_0;
		Tsubdiag_k(0) = beta_0;

		//Compute next vector of Q_k: q_1
		Q_k.col(1) = r / beta_0;

		//Start the iterations
		for (int k = 1; k < max_it; ++k) {

			//Get previous values
			q_prev = Q_k.col(k - 1);
			q_curr = Q_k.col(k);
			beta_prev = Tsubdiag_k(k - 1);

			//Compute next alpha value
			v_1 = w_plus_d_inv_sqrt_inv.cwiseProduct(q_prev);
			v_2 = B_rm.triangularView<Eigen::UpLoType::UnitLower>().solve(v_1);
			v_3 = w.cwiseProduct(v_2);
			v_4 = B_rm.transpose().triangularView<Eigen::UpLoType::UnitUpper>().solve(v_3);
			r = w_plus_d_inv_sqrt_inv.cwiseProduct(v_4);
			r -= beta_prev * q_prev;
			alpha_curr = q_curr.dot(r);

			//Store alpha_curr
			Tdiag_k(k) = alpha_curr;
			final_rank += 1;

			if ((k + 1) < max_it) {

				//Compute next residual
				r -= alpha_curr * q_curr;

				//Full reorthogonalization: r = r - Q_k (Q_k' r)
				r -= Q_k(Eigen::all, Eigen::seq(0, k)) * (Q_k(Eigen::all, Eigen::seq(0, k)).transpose() * r);

				//Compute next beta value
				beta_curr = r.norm();
				Tsubdiag_k(k) = beta_curr;

				r /= beta_curr;

				//More reorthogonalizations if necessary
				inner_products = Q_k(Eigen::all, Eigen::seq(0, k)).transpose() * r;
				could_reorthogonalize = false;
				for (int l = 0; l < 10; ++l) {
					if ((inner_products.array() < tol).all()) {
						could_reorthogonalize = true;
						break;
					}
					r -= Q_k(Eigen::all, Eigen::seq(0, k)) * (Q_k(Eigen::all, Eigen::seq(0, k)).transpose() * r);
					r /= r.norm();
					inner_products = Q_k(Eigen::all, Eigen::seq(0, k)).transpose() * r;
				}

				//Store next vector of Q_k
				Q_k.col(k + 1) = r;

				if (abs(beta_curr) < 1e-6 || !could_reorthogonalize) {
					Log::REInfo("Lanczos tridiagonalization finished beta");
					if (std::abs(beta_curr) < 1e-6) {
						Log::REInfo("Lanczos stopped: beta_curr too small (%g)", beta_curr);
					}
					if (!could_reorthogonalize) {
						Log::REInfo("Lanczos stopped: reorthogonalization failed.");
					}
					break;
				}
			}
		}

		//Resize Q_k, Tdiag_k, Tsubdiag_k
		max_it = final_rank;
		Q_k.conservativeResize(num_data, final_rank);
		Tdiag_k.conservativeResize(final_rank, 1);
		Tsubdiag_k.conservativeResize(final_rank - 1, 1);

		// Print final rank
		Log::REInfo("Lanczos tridiagonalization finished with final rank %i", final_rank);
		std::stringstream ss_diag, ss_sub;
		ss_diag << Tdiag_k.transpose();
		ss_sub << Tsubdiag_k.transpose();
		Log::REInfo("Diagonal T: %s", ss_diag.str().c_str());
		Log::REInfo("Subdiagonal T: %s", ss_sub.str().c_str());
	} // end LanczosTridiagVecchiaLaplaceNoPreconditioner
}

