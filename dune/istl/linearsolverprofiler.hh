// SPDX-FileCopyrightText: Copyright © DUNE Project contributors, see file LICENSE.md in module root
// SPDX-License-Identifier: LicenseRef-GPL-2.0-only-with-DUNE-exception
#ifndef DUNE_ISTL_LINEAR_SOLVER_PROFILER_HH
#define DUNE_ISTL_LINEAR_SOLVER_PROFILER_HH

#include <chrono>
#include <string>
#include <vector>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <memory>
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <filesystem>
#include <sys/resource.h>

#if defined(HAVE_MPI)
#include <mpi.h>
#endif

namespace Dune {

class LinearSolverProfiler {
public:
    // Configuration
    bool enabled = false;
    bool detailed = false;
    bool print_summary = true;
    std::string output_file = "";
    std::string base_output_dir = "benchmark_runs";
    std::string start_timestamp = "";
    std::string session_dir = "";
    int mpi_rank = 0;
    int mpi_size = 1;
    bool finalized = false;
    bool session_initialized = false;

    // --- Matrix & Hardware Metrics ---
    uint64_t matrix_rows = 0;
    uint64_t matrix_cols = 0;
    uint64_t matrix_nnz = 0;
    uint64_t matrix_block_nnz = 0;

    // --- BiCGSTAB Solver Level Metrics ---
    uint64_t num_solves = 0;
    uint64_t total_iterations = 0;
    double total_solve_time = 0.0;

    double spmv_time = 0.0;
    uint64_t spmv_calls = 0;

    double prec_apply_time = 0.0;
    uint64_t prec_apply_calls = 0;

    double dot_prod_time = 0.0;
    uint64_t dot_prod_calls = 0;

    double vec_ops_time = 0.0;
    uint64_t vec_ops_calls = 0;

    // --- ILU(0) Preconditioner Metrics ---
    double ilu0_setup_time = 0.0;
    uint64_t ilu0_setup_calls = 0;

    double ilu0_lower_solve_time = 0.0;
    uint64_t ilu0_lower_solve_calls = 0;

    double ilu0_upper_solve_time = 0.0;
    uint64_t ilu0_upper_solve_calls = 0;

    double ilu0_reorder_time = 0.0;
    uint64_t ilu0_reorder_calls = 0;

    double ilu0_comm_time = 0.0;
    uint64_t ilu0_comm_calls = 0;

    double ilu0_relax_time = 0.0;
    uint64_t ilu0_relax_calls = 0;

    // --- CPR Preconditioner Metrics ---
    double cpr_setup_time = 0.0;
    uint64_t cpr_setup_calls = 0;

    double cpr_presmooth_time = 0.0;
    uint64_t cpr_presmooth_calls = 0;

    double cpr_restriction_time = 0.0;
    uint64_t cpr_restriction_calls = 0;

    double cpr_coarse_solve_time = 0.0;
    uint64_t cpr_coarse_solve_calls = 0;

    double cpr_prolongation_time = 0.0;
    uint64_t cpr_prolongation_calls = 0;

    double cpr_postsmooth_time = 0.0;
    uint64_t cpr_postsmooth_calls = 0;

    // Detailed Per-Iteration and Per-Solve Records
    struct IterationRecord {
        uint64_t solve_id = 0;
        double iteration = 0.0;
        double residual_norm = 0.0;
        double spmv_time = 0.0;
        double prec_time = 0.0;
        double dot_time = 0.0;
        double vec_time = 0.0;
    };

    struct SolveRecord {
        uint64_t solve_id = 0;
        uint64_t iterations = 0;
        double solve_wall_time = 0.0;
        double initial_residual = 0.0;
        double final_residual = 0.0;
        bool converged = false;
    };

    // Pod packet for gathering all ranks' metrics into a single unified file
    struct RankMetricsPacket {
        int rank = 0;
        uint64_t num_solves = 0;
        uint64_t total_iterations = 0;
        double total_solve_time = 0.0;
        double spmv_time = 0.0;
        uint64_t spmv_calls = 0;
        double prec_apply_time = 0.0;
        uint64_t prec_apply_calls = 0;
        double dot_prod_time = 0.0;
        uint64_t dot_prod_calls = 0;
        double vec_ops_time = 0.0;
        uint64_t vec_ops_calls = 0;
        double ilu0_setup_time = 0.0;
        uint64_t ilu0_setup_calls = 0;
        double ilu0_lower_solve_time = 0.0;
        uint64_t ilu0_lower_solve_calls = 0;
        double ilu0_upper_solve_time = 0.0;
        uint64_t ilu0_upper_solve_calls = 0;
        double ilu0_reorder_time = 0.0;
        uint64_t ilu0_reorder_calls = 0;
        double ilu0_comm_time = 0.0;
        uint64_t ilu0_comm_calls = 0;
        double ilu0_relax_time = 0.0;
        uint64_t ilu0_relax_calls = 0;
        double cpr_setup_time = 0.0;
        uint64_t cpr_setup_calls = 0;
        double cpr_presmooth_time = 0.0;
        uint64_t cpr_presmooth_calls = 0;
        double cpr_restriction_time = 0.0;
        uint64_t cpr_restriction_calls = 0;
        double cpr_coarse_solve_time = 0.0;
        uint64_t cpr_coarse_solve_calls = 0;
        double cpr_prolongation_time = 0.0;
        uint64_t cpr_prolongation_calls = 0;
        double cpr_postsmooth_time = 0.0;
        uint64_t cpr_postsmooth_calls = 0;
        uint64_t matrix_rows = 0;
        uint64_t matrix_cols = 0;
        uint64_t matrix_nnz = 0;
        double peak_memory_mb = 0.0;
        double spmv_bandwidth_gbs = 0.0;
    };

    RankMetricsPacket pack() const {
        RankMetricsPacket p;
        p.rank = mpi_rank;
        p.num_solves = num_solves;
        p.total_iterations = total_iterations;
        p.total_solve_time = total_solve_time;
        p.spmv_time = spmv_time;
        p.spmv_calls = spmv_calls;
        p.prec_apply_time = prec_apply_time;
        p.prec_apply_calls = prec_apply_calls;
        p.dot_prod_time = dot_prod_time;
        p.dot_prod_calls = dot_prod_calls;
        p.vec_ops_time = vec_ops_time;
        p.vec_ops_calls = vec_ops_calls;
        p.ilu0_setup_time = ilu0_setup_time;
        p.ilu0_setup_calls = ilu0_setup_calls;
        p.ilu0_lower_solve_time = ilu0_lower_solve_time;
        p.ilu0_lower_solve_calls = ilu0_lower_solve_calls;
        p.ilu0_upper_solve_time = ilu0_upper_solve_time;
        p.ilu0_upper_solve_calls = ilu0_upper_solve_calls;
        p.ilu0_reorder_time = ilu0_reorder_time;
        p.ilu0_reorder_calls = ilu0_reorder_calls;
        p.ilu0_comm_time = ilu0_comm_time;
        p.ilu0_comm_calls = ilu0_comm_calls;
        p.ilu0_relax_time = ilu0_relax_time;
        p.ilu0_relax_calls = ilu0_relax_calls;
        p.cpr_setup_time = cpr_setup_time;
        p.cpr_setup_calls = cpr_setup_calls;
        p.cpr_presmooth_time = cpr_presmooth_time;
        p.cpr_presmooth_calls = cpr_presmooth_calls;
        p.cpr_restriction_time = cpr_restriction_time;
        p.cpr_restriction_calls = cpr_restriction_calls;
        p.cpr_coarse_solve_time = cpr_coarse_solve_time;
        p.cpr_coarse_solve_calls = cpr_coarse_solve_calls;
        p.cpr_prolongation_time = cpr_prolongation_time;
        p.cpr_prolongation_calls = cpr_prolongation_calls;
        p.cpr_postsmooth_time = cpr_postsmooth_time;
        p.cpr_postsmooth_calls = cpr_postsmooth_calls;
        p.matrix_rows = matrix_rows;
        p.matrix_cols = matrix_cols;
        p.matrix_nnz = matrix_nnz;
        p.peak_memory_mb = getPeakMemoryMB();
        p.spmv_bandwidth_gbs = getSpmvBandwidthGBS();
        return p;
    }

    std::vector<IterationRecord> iteration_history;
    std::vector<SolveRecord> solve_history;

    // Singleton instance access
    static LinearSolverProfiler& instance() {
        static LinearSolverProfiler profiler;
        return profiler;
    }

    ~LinearSolverProfiler() {
        finalize();
    }

    void initSession(const std::string& base_dir = "") {
        if (session_initialized) {
            return;
        }
        if (!base_dir.empty()) {
            base_output_dir = base_dir;
        }

#if defined(HAVE_MPI)
        int mpi_initialized = 0;
        int mpi_finalized_flag = 0;
        MPI_Initialized(&mpi_initialized);
        MPI_Finalized(&mpi_finalized_flag);
        if (mpi_initialized && !mpi_finalized_flag) {
            MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
            MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
        }
#endif

        // Generate timestamp representing when the simulation started
        if (mpi_rank == 0 && start_timestamp.empty()) {
            auto now = std::chrono::system_clock::now();
            auto in_time_t = std::chrono::system_clock::to_time_t(now);
            std::tm tm_buf;
            localtime_r(&in_time_t, &tm_buf);
            std::ostringstream ss;
            ss << std::put_time(&tm_buf, "%Y%m%d_%H%M%S");
            start_timestamp = ss.str();
        }

#if defined(HAVE_MPI)
        if (mpi_size > 1 && mpi_initialized && !mpi_finalized_flag) {
            char ts_buf[64] = {0};
            if (mpi_rank == 0) {
                std::strncpy(ts_buf, start_timestamp.c_str(), sizeof(ts_buf) - 1);
            }
            MPI_Bcast(ts_buf, sizeof(ts_buf), MPI_CHAR, 0, MPI_COMM_WORLD);
            start_timestamp = std::string(ts_buf);
        }
#endif

        session_dir = base_output_dir + "/run_" + start_timestamp;

        if (mpi_rank == 0) {
            std::error_code ec;
            std::filesystem::create_directories(session_dir, ec);
        }

#if defined(HAVE_MPI)
        if (mpi_size > 1 && mpi_initialized && !mpi_finalized_flag) {
            MPI_Barrier(MPI_COMM_WORLD);
        }
#endif
        session_initialized = true;
    }

    void recordMatrix(uint64_t rows, uint64_t cols, uint64_t nnz, uint64_t block_nnz) {
        matrix_rows = rows;
        matrix_cols = cols;
        matrix_nnz = nnz;
        matrix_block_nnz = block_nnz;
    }

    static double getPeakMemoryMB() {
        struct rusage r_usage;
        if (getrusage(RUSAGE_SELF, &r_usage) == 0) {
            return static_cast<double>(r_usage.ru_maxrss) / 1024.0;
        }
        return 0.0;
    }

    double getSpmvBandwidthGBS() const {
        if (spmv_time <= 0.0 || spmv_calls == 0 || matrix_nnz == 0) {
            return 0.0;
        }
        const double bytes_per_spmv = static_cast<double>(matrix_nnz) * 8.0
                                    + static_cast<double>(matrix_block_nnz) * 4.0
                                    + static_cast<double>(matrix_cols + 2 * matrix_rows) * 8.0;
        return (static_cast<double>(spmv_calls) * bytes_per_spmv) / (spmv_time * 1e9);
    }

    void reset() {
        num_solves = 0;
        total_iterations = 0;
        total_solve_time = 0.0;
        spmv_time = 0.0;
        spmv_calls = 0;
        prec_apply_time = 0.0;
        prec_apply_calls = 0;
        dot_prod_time = 0.0;
        dot_prod_calls = 0;
        vec_ops_time = 0.0;
        vec_ops_calls = 0;

        ilu0_setup_time = 0.0;
        ilu0_setup_calls = 0;
        ilu0_lower_solve_time = 0.0;
        ilu0_lower_solve_calls = 0;
        ilu0_upper_solve_time = 0.0;
        ilu0_upper_solve_calls = 0;
        ilu0_reorder_time = 0.0;
        ilu0_reorder_calls = 0;
        ilu0_comm_time = 0.0;
        ilu0_comm_calls = 0;
        ilu0_relax_time = 0.0;
        ilu0_relax_calls = 0;

        cpr_setup_time = 0.0;
        cpr_setup_calls = 0;
        cpr_presmooth_time = 0.0;
        cpr_presmooth_calls = 0;
        cpr_restriction_time = 0.0;
        cpr_restriction_calls = 0;
        cpr_coarse_solve_time = 0.0;
        cpr_coarse_solve_calls = 0;
        cpr_prolongation_time = 0.0;
        cpr_prolongation_calls = 0;
        cpr_postsmooth_time = 0.0;
        cpr_postsmooth_calls = 0;

        matrix_rows = 0;
        matrix_cols = 0;
        matrix_nnz = 0;
        matrix_block_nnz = 0;

        iteration_history.clear();
        solve_history.clear();
        finalized = false;
        session_initialized = false;
        session_dir = "";
        start_timestamp = "";
    }

    void printSummary(std::ostream& out = std::cout) const {
        if (num_solves == 0 && total_solve_time == 0.0) {
            return;
        }

        const double avg_iter_per_solve = (num_solves > 0) ? (static_cast<double>(total_iterations) / num_solves) : 0.0;
        const double avg_time_per_solve_ms = (num_solves > 0) ? ((total_solve_time / num_solves) * 1000.0) : 0.0;
        const double avg_time_per_iter_ms = (total_iterations > 0) ? ((total_solve_time / total_iterations) * 1000.0) : 0.0;

        auto pct = [this](double t) -> double {
            return (total_solve_time > 0.0) ? (t / total_solve_time * 100.0) : 0.0;
        };
        auto avg_call_ms = [](double t, uint64_t c) -> double {
            return (c > 0) ? ((t / c) * 1000.0) : 0.0;
        };
        auto ratio_to_solve = [this](double setup_t) -> std::string {
            if (total_solve_time <= 0.0) return "-";
            std::ostringstream ss;
            ss << std::fixed << std::setprecision(1) << (setup_t / total_solve_time * 100.0) << "% of solve";
            return ss.str();
        };

        out << "\n================================================================================\n";
        out << "                   LINEAR SOLVER HPC PROFILING REPORT\n";
        if (mpi_size > 1) {
            out << "                      (MPI Rank: " << mpi_rank << " / " << mpi_size << ")\n";
        }
        out << "================================================================================\n";
        out << std::left << std::setw(32) << "Total Linear Solves:" << num_solves << "\n";
        out << std::left << std::setw(32) << "Total Linear Iterations:" << total_iterations
            << " (Avg: " << std::fixed << std::setprecision(2) << avg_iter_per_solve << " iter/solve)\n";
        out << std::left << std::setw(32) << "Total Linear Wall Time:" << std::fixed << std::setprecision(4)
            << total_solve_time << " s\n";
        out << std::left << std::setw(32) << "Avg Time per Solve:" << std::fixed << std::setprecision(4)
            << avg_time_per_solve_ms << " ms\n";
        out << std::left << std::setw(32) << "Avg Time per Iteration:" << std::fixed << std::setprecision(4)
            << avg_time_per_iter_ms << " ms\n";

        if (matrix_rows > 0) {
            double density = (matrix_rows > 0 && matrix_cols > 0)
                             ? (static_cast<double>(matrix_nnz) / (static_cast<double>(matrix_rows) * matrix_cols) * 100.0)
                             : 0.0;
            out << std::left << std::setw(32) << "Local Matrix Dimensions:" << matrix_rows << " x " << matrix_cols << "\n";
            out << std::left << std::setw(32) << "Matrix Non-Zeros (NNZ):" << matrix_nnz
                << " (Density: " << std::fixed << std::setprecision(3) << density << "%)\n";
        }
        const double peak_rss = getPeakMemoryMB();
        if (peak_rss > 0.0) {
            out << std::left << std::setw(32) << "Peak Process Memory (RSS):" << std::fixed << std::setprecision(1) << peak_rss << " MB\n";
        }
        const double spmv_bw = getSpmvBandwidthGBS();
        if (spmv_bw > 0.0) {
            out << std::left << std::setw(32) << "Sustained SpMV Bandwidth:" << std::fixed << std::setprecision(2) << spmv_bw << " GB/s\n";
        }

        out << "--------------------------------------------------------------------------------\n";
        out << std::left << std::setw(28) << "Phase / Kernel"
            << std::right << std::setw(12) << "Time (s)"
            << std::setw(12) << "Calls"
            << std::setw(16) << "Avg/Call (ms)"
            << std::setw(14) << "% Solver\n";
        out << "--------------------------------------------------------------------------------\n";

        // BiCGSTAB components
        out << std::left << std::setw(28) << "BiCGSTAB SpMV:"
            << std::right << std::fixed << std::setprecision(4)
            << std::setw(12) << spmv_time
            << std::setw(12) << spmv_calls
            << std::setw(16) << avg_call_ms(spmv_time, spmv_calls)
            << std::setw(13) << std::setprecision(1) << pct(spmv_time) << "%\n";

        out << std::left << std::setw(28) << "BiCGSTAB Precond Apply:"
            << std::right << std::fixed << std::setprecision(4)
            << std::setw(12) << prec_apply_time
            << std::setw(12) << prec_apply_calls
            << std::setw(16) << avg_call_ms(prec_apply_time, prec_apply_calls)
            << std::setw(13) << std::setprecision(1) << pct(prec_apply_time) << "%\n";

        out << std::left << std::setw(28) << "BiCGSTAB Dot Products:"
            << std::right << std::fixed << std::setprecision(4)
            << std::setw(12) << dot_prod_time
            << std::setw(12) << dot_prod_calls
            << std::setw(16) << avg_call_ms(dot_prod_time, dot_prod_calls)
            << std::setw(13) << std::setprecision(1) << pct(dot_prod_time) << "%\n";

        out << std::left << std::setw(28) << "BiCGSTAB Vector Ops (AXPY):"
            << std::right << std::fixed << std::setprecision(4)
            << std::setw(12) << vec_ops_time
            << std::setw(12) << vec_ops_calls
            << std::setw(16) << avg_call_ms(vec_ops_time, vec_ops_calls)
            << std::setw(13) << std::setprecision(1) << pct(vec_ops_time) << "%\n";

        // Preconditioner Level: ILU(0)
        if (ilu0_setup_calls > 0 || ilu0_lower_solve_calls > 0 || ilu0_upper_solve_calls > 0) {
            out << "--------------------------------------------------------------------------------\n";
            out << "[ILU(0) Preconditioner Breakdown]\n";
            out << std::left << std::setw(28) << "ILU(0) Factorization Setup:"
                << std::right << std::fixed << std::setprecision(4)
                << std::setw(12) << ilu0_setup_time
                << std::setw(12) << ilu0_setup_calls
                << std::setw(16) << avg_call_ms(ilu0_setup_time, ilu0_setup_calls)
                << std::setw(14) << ("[" + ratio_to_solve(ilu0_setup_time) + "]") << "\n";

            out << std::left << std::setw(28) << "ILU(0) Lower Tri Solve (L):"
                << std::right << std::fixed << std::setprecision(4)
                << std::setw(12) << ilu0_lower_solve_time
                << std::setw(12) << ilu0_lower_solve_calls
                << std::setw(16) << avg_call_ms(ilu0_lower_solve_time, ilu0_lower_solve_calls)
                << std::setw(13) << std::setprecision(1) << pct(ilu0_lower_solve_time) << "%\n";

            out << std::left << std::setw(28) << "ILU(0) Upper Tri Solve (U):"
                << std::right << std::fixed << std::setprecision(4)
                << std::setw(12) << ilu0_upper_solve_time
                << std::setw(12) << ilu0_upper_solve_calls
                << std::setw(16) << avg_call_ms(ilu0_upper_solve_time, ilu0_upper_solve_calls)
                << std::setw(13) << std::setprecision(1) << pct(ilu0_upper_solve_time) << "%\n";

            if (ilu0_reorder_calls > 0) {
                out << std::left << std::setw(28) << "ILU(0) Vector Reordering:"
                    << std::right << std::fixed << std::setprecision(4)
                    << std::setw(12) << ilu0_reorder_time
                    << std::setw(12) << ilu0_reorder_calls
                    << std::setw(16) << avg_call_ms(ilu0_reorder_time, ilu0_reorder_calls)
                    << std::setw(13) << std::setprecision(1) << pct(ilu0_reorder_time) << "%\n";
            }
            if (ilu0_comm_calls > 0) {
                out << std::left << std::setw(28) << "ILU(0) Halo Communication:"
                    << std::right << std::fixed << std::setprecision(4)
                    << std::setw(12) << ilu0_comm_time
                    << std::setw(12) << ilu0_comm_calls
                    << std::setw(16) << avg_call_ms(ilu0_comm_time, ilu0_comm_calls)
                    << std::setw(13) << std::setprecision(1) << pct(ilu0_comm_time) << "%\n";
            }
            if (ilu0_relax_calls > 0) {
                out << std::left << std::setw(28) << "ILU(0) Vector Relaxation:"
                    << std::right << std::fixed << std::setprecision(4)
                    << std::setw(12) << ilu0_relax_time
                    << std::setw(12) << ilu0_relax_calls
                    << std::setw(16) << avg_call_ms(ilu0_relax_time, ilu0_relax_calls)
                    << std::setw(13) << std::setprecision(1) << pct(ilu0_relax_time) << "%\n";
            }
        }

        // Preconditioner Level: CPR
        if (cpr_setup_calls > 0 || cpr_coarse_solve_calls > 0 || cpr_presmooth_calls > 0) {
            out << "--------------------------------------------------------------------------------\n";
            out << "[CPR Preconditioner Breakdown]\n";
            out << std::left << std::setw(28) << "CPR Coarse System Setup:"
                << std::right << std::fixed << std::setprecision(4)
                << std::setw(12) << cpr_setup_time
                << std::setw(12) << cpr_setup_calls
                << std::setw(16) << avg_call_ms(cpr_setup_time, cpr_setup_calls)
                << std::setw(14) << ("[" + ratio_to_solve(cpr_setup_time) + "]") << "\n";

            out << std::left << std::setw(28) << "CPR Pre-smoothing:"
                << std::right << std::fixed << std::setprecision(4)
                << std::setw(12) << cpr_presmooth_time
                << std::setw(12) << cpr_presmooth_calls
                << std::setw(16) << avg_call_ms(cpr_presmooth_time, cpr_presmooth_calls)
                << std::setw(13) << std::setprecision(1) << pct(cpr_presmooth_time) << "%\n";

            out << std::left << std::setw(28) << "CPR Restriction (Fine->Coarse):"
                << std::right << std::fixed << std::setprecision(4)
                << std::setw(12) << cpr_restriction_time
                << std::setw(12) << cpr_restriction_calls
                << std::setw(16) << avg_call_ms(cpr_restriction_time, cpr_restriction_calls)
                << std::setw(13) << std::setprecision(1) << pct(cpr_restriction_time) << "%\n";

            out << std::left << std::setw(28) << "CPR Coarse AMG Solve:"
                << std::right << std::fixed << std::setprecision(4)
                << std::setw(12) << cpr_coarse_solve_time
                << std::setw(12) << cpr_coarse_solve_calls
                << std::setw(16) << avg_call_ms(cpr_coarse_solve_time, cpr_coarse_solve_calls)
                << std::setw(13) << std::setprecision(1) << pct(cpr_coarse_solve_time) << "%\n";

            out << std::left << std::setw(28) << "CPR Prolongation (Coarse->Fine):"
                << std::right << std::fixed << std::setprecision(4)
                << std::setw(12) << cpr_prolongation_time
                << std::setw(12) << cpr_prolongation_calls
                << std::setw(16) << avg_call_ms(cpr_prolongation_time, cpr_prolongation_calls)
                << std::setw(13) << std::setprecision(1) << pct(cpr_prolongation_time) << "%\n";

            out << std::left << std::setw(28) << "CPR Post-smoothing:"
                << std::right << std::fixed << std::setprecision(4)
                << std::setw(12) << cpr_postsmooth_time
                << std::setw(12) << cpr_postsmooth_calls
                << std::setw(16) << avg_call_ms(cpr_postsmooth_time, cpr_postsmooth_calls)
                << std::setw(13) << std::setprecision(1) << pct(cpr_postsmooth_time) << "%\n";
        }

        if (!session_dir.empty()) {
            out << "--------------------------------------------------------------------------------\n";
            out << "Benchmark files saved to: " << session_dir << "\n";
            out << "  • Unified summary (all ranks): " << session_dir << "/all_ranks_summary.csv\n";
            out << "  • Individual rank files:      " << session_dir << "/rank_" << mpi_rank << ".csv\n";
        }
        out << "================================================================================\n\n";
    }

#if defined(HAVE_MPI)
    void printMpiSummary(std::ostream& out,
                         const std::vector<double>& min_times,
                         const std::vector<double>& max_times,
                         const std::vector<double>& sum_times,
                         const std::vector<std::string>& labels) const
    {
        out << "================================================================================\n";
        out << "          MPI CROSS-RANK SCALABILITY & LOAD IMBALANCE REPORT (" << mpi_size << " Ranks)\n";
        out << "================================================================================\n";
        out << std::left << std::setw(28) << "Phase / Kernel"
            << std::right << std::setw(12) << "Min Time (s)"
            << std::setw(13) << "Avg Time (s)"
            << std::setw(13) << "Max Time (s)"
            << std::setw(14) << "Imbalance (%)\n";
        out << "--------------------------------------------------------------------------------\n";

        for (size_t i = 0; i < labels.size(); ++i) {
            double min_t = min_times[i];
            double max_t = max_times[i];
            double avg_t = (mpi_size > 0) ? (sum_times[i] / mpi_size) : 0.0;
            double imbalance = (avg_t > 0.0) ? ((max_t - avg_t) / avg_t * 100.0) : 0.0;

            if (max_t > 0.0) {
                out << std::left << std::setw(28) << labels[i]
                    << std::right << std::fixed << std::setprecision(4)
                    << std::setw(12) << min_t
                    << std::setw(13) << avg_t
                    << std::setw(13) << max_t
                    << std::setw(13) << std::setprecision(1) << imbalance << "%\n";
            }
        }
        out << "================================================================================\n\n";
    }
#endif

    static void writePacketToCsv(std::ostream& csv, const RankMetricsPacket& p) {
        auto pct = [&](double t) -> double {
            return (p.total_solve_time > 0.0) ? (t / p.total_solve_time * 100.0) : 0.0;
        };
        auto avg_call_ms = [](double t, uint64_t c) -> double {
            return (c > 0) ? ((t / c) * 1000.0) : 0.0;
        };

        auto write_row = [&](const std::string& metric, double t, uint64_t c, double p_solve) {
            csv << p.rank << "," << metric << "," << t << "," << c << ","
                << avg_call_ms(t, c) << "," << p_solve << ","
                << p.matrix_rows << "," << p.matrix_nnz << ","
                << p.peak_memory_mb << "," << p.spmv_bandwidth_gbs << "\n";
        };

        write_row("total_solve", p.total_solve_time, p.num_solves, 100.0);
        write_row("spmv", p.spmv_time, p.spmv_calls, pct(p.spmv_time));
        write_row("prec_apply", p.prec_apply_time, p.prec_apply_calls, pct(p.prec_apply_time));
        write_row("dot_products", p.dot_prod_time, p.dot_prod_calls, pct(p.dot_prod_time));
        write_row("vec_ops", p.vec_ops_time, p.vec_ops_calls, pct(p.vec_ops_time));
        write_row("ilu0_setup", p.ilu0_setup_time, p.ilu0_setup_calls, 0.0);
        write_row("ilu0_lower_solve", p.ilu0_lower_solve_time, p.ilu0_lower_solve_calls, pct(p.ilu0_lower_solve_time));
        write_row("ilu0_upper_solve", p.ilu0_upper_solve_time, p.ilu0_upper_solve_calls, pct(p.ilu0_upper_solve_time));
        write_row("ilu0_reorder", p.ilu0_reorder_time, p.ilu0_reorder_calls, pct(p.ilu0_reorder_time));
        write_row("ilu0_comm", p.ilu0_comm_time, p.ilu0_comm_calls, pct(p.ilu0_comm_time));
        write_row("ilu0_relax", p.ilu0_relax_time, p.ilu0_relax_calls, pct(p.ilu0_relax_time));
        write_row("cpr_setup", p.cpr_setup_time, p.cpr_setup_calls, 0.0);
        write_row("cpr_presmooth", p.cpr_presmooth_time, p.cpr_presmooth_calls, pct(p.cpr_presmooth_time));
        write_row("cpr_restriction", p.cpr_restriction_time, p.cpr_restriction_calls, pct(p.cpr_restriction_time));
        write_row("cpr_coarse_solve", p.cpr_coarse_solve_time, p.cpr_coarse_solve_calls, pct(p.cpr_coarse_solve_time));
        write_row("cpr_prolongation", p.cpr_prolongation_time, p.cpr_prolongation_calls, pct(p.cpr_prolongation_time));
        write_row("cpr_postsmooth", p.cpr_postsmooth_time, p.cpr_postsmooth_calls, pct(p.cpr_postsmooth_time));
    }

    void exportSessionFiles() const {
        if (session_dir.empty()) {
            return;
        }

        // 1. Every rank writes its own separate CSV file inside session_dir
        std::string rank_fname = session_dir + "/rank_" + std::to_string(mpi_rank) + ".csv";
        std::ofstream rcsv(rank_fname);
        if (rcsv.is_open()) {
            rcsv << "rank,metric,time_seconds,call_count,avg_time_ms,pct_linear_solver,matrix_rows,matrix_nnz,peak_memory_mb,spmv_bandwidth_gbs\n";
            writePacketToCsv(rcsv, pack());
            rcsv.close();
        }

        // 2. Optional detailed iteration history for each rank
        if (detailed && !iteration_history.empty()) {
            std::string iter_fname = session_dir + "/iterations_rank_" + std::to_string(mpi_rank) + ".csv";
            std::ofstream icsv(iter_fname);
            if (icsv.is_open()) {
                icsv << "rank,solve_id,iteration,residual_norm,spmv_time_s,prec_time_s,dot_time_s,vec_time_s\n";
                for (const auto& rec : iteration_history) {
                    icsv << mpi_rank << "," << rec.solve_id << "," << rec.iteration << "," << rec.residual_norm << ","
                         << rec.spmv_time << "," << rec.prec_time << "," << rec.dot_time << "," << rec.vec_time << "\n";
                }
                icsv.close();
            }
        }
    }

    void finalize() {
        if (finalized || !enabled) {
            return;
        }

        if (!session_initialized) {
            initSession(base_output_dir);
        }

        // Save this rank's individual file
        exportSessionFiles();

#if defined(HAVE_MPI)
        int mpi_initialized = 0;
        int mpi_finalized_flag = 0;
        MPI_Initialized(&mpi_initialized);
        MPI_Finalized(&mpi_finalized_flag);
        if (mpi_initialized && !mpi_finalized_flag) {
            MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
            MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
        }

        std::vector<std::string> mpi_labels = {
            "BiCGSTAB SpMV",
            "BiCGSTAB Precond Apply",
            "BiCGSTAB Dot Products",
            "BiCGSTAB Vector Ops",
            "Total Linear Wall Time",
            "ILU(0) Lower Tri Solve",
            "ILU(0) Upper Tri Solve",
            "ILU(0) Halo Communication",
            "CPR Coarse AMG Solve",
            "CPR Post-smoothing"
        };
        std::vector<double> local_times = {
            spmv_time,
            prec_apply_time,
            dot_prod_time,
            vec_ops_time,
            total_solve_time,
            ilu0_lower_solve_time,
            ilu0_upper_solve_time,
            ilu0_comm_time,
            cpr_coarse_solve_time,
            cpr_postsmooth_time
        };
        const int num_metrics = static_cast<int>(local_times.size());
        std::vector<double> min_times(num_metrics, 0.0);
        std::vector<double> max_times(num_metrics, 0.0);
        std::vector<double> sum_times(num_metrics, 0.0);

        RankMetricsPacket my_packet = pack();
        std::vector<RankMetricsPacket> all_packets(mpi_size > 0 ? mpi_size : 1);

        if (mpi_initialized && !mpi_finalized_flag && mpi_size > 1) {
            MPI_Reduce(local_times.data(), min_times.data(), num_metrics, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
            MPI_Reduce(local_times.data(), max_times.data(), num_metrics, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
            MPI_Reduce(local_times.data(), sum_times.data(), num_metrics, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

            // Gather all rank packets to Rank 0.
            // Allocate backing storage on all ranks to prevent memory corruption in
            // tree-based / hierarchical collective gather algorithms (e.g. OpenMPI / PRRTE).
            MPI_Gather(&my_packet, sizeof(RankMetricsPacket), MPI_BYTE,
                       all_packets.data(), sizeof(RankMetricsPacket), MPI_BYTE,
                       0, MPI_COMM_WORLD);
        }
#endif

        if (mpi_rank == 0) {
            // Write unified all-ranks file
            std::string all_fname = session_dir + "/all_ranks_summary.csv";
            std::ofstream all_csv(all_fname);
            if (all_csv.is_open()) {
                all_csv << "rank,metric,time_seconds,call_count,avg_time_ms,pct_linear_solver,matrix_rows,matrix_nnz,peak_memory_mb,spmv_bandwidth_gbs\n";
#if defined(HAVE_MPI)
                if (mpi_size > 1 && !all_packets.empty()) {
                    for (int r = 0; r < mpi_size; ++r) {
                        writePacketToCsv(all_csv, all_packets[r]);
                    }
                } else
#endif
                {
                    writePacketToCsv(all_csv, pack());
                }
                all_csv.close();
            }

            // Save text copy of the summary report
            std::string report_txt_fname = session_dir + "/summary_report.txt";
            std::ofstream rtxt(report_txt_fname);
            if (rtxt.is_open()) {
                printSummary(rtxt);
#if defined(HAVE_MPI)
                if (mpi_initialized && !mpi_finalized_flag && mpi_size > 1) {
                    printMpiSummary(rtxt, min_times, max_times, sum_times, mpi_labels);
                }
#endif
                rtxt.close();
            }

            // Also output to legacy output_file if explicitly specified
            if (!output_file.empty()) {
                std::ofstream legacy_csv(output_file);
                if (legacy_csv.is_open()) {
                    legacy_csv << "rank,metric,time_seconds,call_count,avg_time_ms,pct_linear_solver,matrix_rows,matrix_nnz,peak_memory_mb,spmv_bandwidth_gbs\n";
#if defined(HAVE_MPI)
                    if (mpi_size > 1 && !all_packets.empty()) {
                        for (int r = 0; r < mpi_size; ++r) {
                            writePacketToCsv(legacy_csv, all_packets[r]);
                        }
                    } else
#endif
                    {
                        writePacketToCsv(legacy_csv, pack());
                    }
                    legacy_csv.close();
                }
            }

            // Print terminal report
            if (print_summary) {
                printSummary(std::cout);
#if defined(HAVE_MPI)
                if (mpi_initialized && !mpi_finalized_flag && mpi_size > 1) {
                    printMpiSummary(std::cout, min_times, max_times, sum_times, mpi_labels);
                }
#endif
            }
        }

        finalized = true;
    }
};

// Zero-overhead RAII Scoped Kernel Timer
class ScopedKernelTimer {
    bool active_;
    double& time_acc_;
    uint64_t& call_acc_;
    std::chrono::high_resolution_clock::time_point start_;

public:
    inline ScopedKernelTimer(bool enabled, double& time_acc, uint64_t& call_acc)
        : active_(enabled), time_acc_(time_acc), call_acc_(call_acc)
    {
        if (__builtin_expect(active_, 0)) {
            start_ = std::chrono::high_resolution_clock::now();
        }
    }

    inline ~ScopedKernelTimer() {
        if (__builtin_expect(active_, 0)) {
            auto end = std::chrono::high_resolution_clock::now();
            time_acc_ += std::chrono::duration<double>(end - start_).count();
            ++call_acc_;
        }
    }
};

inline LinearSolverProfiler& getLinearSolverProfiler() {
    return LinearSolverProfiler::instance();
}

} // namespace Dune

#endif // DUNE_ISTL_LINEAR_SOLVER_PROFILER_HH
