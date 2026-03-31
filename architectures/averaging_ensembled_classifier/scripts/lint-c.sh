#!/usr/bin/env bash
# scripts/lint-c.sh — Comprehensive C linting for CPU kernel sources (ADR-015).
#
# Runs available linters against the CPU backend's C code. Uses the Meson-
# generated compile_commands.json for accurate include paths and defines.
#
# Exit codes:
#   0  All checks passed
#   1  Lint errors found
#   2  No linting tools available
#
# Usage:
#   scripts/lint-c.sh              # Run all available linters
#   scripts/lint-c.sh --gcc-only   # GCC warnings only
#   scripts/lint-c.sh --cppcheck   # cppcheck only
#   scripts/lint-c.sh --clang-tidy # clang-tidy only

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARCH_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
KERNEL_DIR="$ARCH_ROOT/src/backends/cpu/kernel_sources"
BUILDDIR="$ARCH_ROOT/builddir"

# Allow tools installed via pip --user
export PATH="/var/data/python/bin:$HOME/.local/bin:$PATH"

C_FILES=(
    "$KERNEL_DIR/cpu_threads.c"
    "$KERNEL_DIR/cpu_kernels.c"
)

H_FILES=(
    "$KERNEL_DIR/cpu_kernels.h"
    "$KERNEL_DIR/cpu_precision.h"
    "$KERNEL_DIR/cpu_simd.h"
    "$KERNEL_DIR/cpu_threads.h"
    "$KERNEL_DIR/cpu_export.h"
)

ERRORS=0
TOOLS_RAN=0

# ─── Colour helpers ───────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BOLD='\033[1m'
NC='\033[0m'

header() { printf "\n${BOLD}═══ %s ═══${NC}\n" "$1"; }
pass()   { printf "${GREEN}✓ %s${NC}\n" "$1"; }
fail()   { printf "${RED}✗ %s${NC}\n" "$1"; }
warn()   { printf "${YELLOW}⚠ %s${NC}\n" "$1"; }

# ─── Mode selection ───────────────────────────────────────────────────────
MODE="${1:-all}"

# ─── GCC comprehensive warnings ──────────────────────────────────────────
run_gcc_lint() {
    if ! command -v gcc &>/dev/null; then
        warn "gcc not found — skipping GCC lint pass"
        return
    fi
    header "GCC Comprehensive Warnings (-Wall -Wextra -fanalyzer)"
    TOOLS_RAN=$((TOOLS_RAN + 1))

    GCC_FLAGS=(
        -fsyntax-only
        -Wall -Wextra -Wpedantic
        -Wconversion -Wsign-conversion
        -Wshadow -Wstrict-prototypes -Wmissing-prototypes
        -Wformat=2 -Wdouble-promotion
        -Wundef -Wpointer-arith -Wcast-align
        -Wnull-dereference -Wwrite-strings
        -fanalyzer
        -I "$KERNEL_DIR"
        -DCPU_KERNELS_BUILDING
        -march=native
    )

    local gcc_out
    if gcc_out=$(gcc "${GCC_FLAGS[@]}" "${C_FILES[@]}" 2>&1); then
        pass "GCC: no warnings"
    else
        echo "$gcc_out"
        # Count actual warnings (exclude notes)
        local warn_count
        warn_count=$(echo "$gcc_out" | grep -c ': warning:' || true)
        local err_count
        err_count=$(echo "$gcc_out" | grep -c ': error:' || true)
        if [ "$err_count" -gt 0 ]; then
            fail "GCC: $err_count error(s), $warn_count warning(s)"
            ERRORS=$((ERRORS + 1))
        elif [ "$warn_count" -gt 0 ]; then
            warn "GCC: $warn_count warning(s)"
        fi
    fi
}

# ─── cppcheck deep analysis ──────────────────────────────────────────────
run_cppcheck() {
    if ! command -v cppcheck &>/dev/null; then
        warn "cppcheck not found — skipping (pip install cppcheck)"
        return
    fi
    header "cppcheck Static Analysis (exhaustive)"
    TOOLS_RAN=$((TOOLS_RAN + 1))

    local supp_file="$ARCH_ROOT/cppcheck.supp"
    local supp_arg=()
    if [ -f "$supp_file" ]; then
        supp_arg=(--suppressions-list="$supp_file")
    fi

    local cppcheck_args=(
        --check-level=exhaustive
        --enable=warning,style,performance,portability
        --inconclusive
        --force
        --std=c11
        --error-exitcode=1
        "${supp_arg[@]}"
        -I "$KERNEL_DIR"
        -DCPU_KERNELS_BUILDING
    )

    # Use compile_commands.json if available for accurate flags
    if [ -f "$BUILDDIR/compile_commands.json" ]; then
        cppcheck_args+=(--project="$BUILDDIR/compile_commands.json")
    else
        cppcheck_args+=("${C_FILES[@]}")
    fi

    local cpp_out
    if cpp_out=$(cppcheck "${cppcheck_args[@]}" 2>&1); then
        pass "cppcheck: no issues"
    else
        # Filter informational messages
        local filtered
        filtered=$(echo "$cpp_out" | grep -v '^\(Checking\|^$\|information:\)' || true)
        if [ -n "$filtered" ]; then
            echo "$filtered"
            fail "cppcheck: issues found"
            ERRORS=$((ERRORS + 1))
        else
            pass "cppcheck: no actionable issues"
        fi
    fi
}

# ─── clang-tidy (when available) ─────────────────────────────────────────
run_clang_tidy() {
    if ! command -v clang-tidy &>/dev/null; then
        warn "clang-tidy not found — skipping (install via package manager)"
        return
    fi
    header "clang-tidy Static Analysis"
    TOOLS_RAN=$((TOOLS_RAN + 1))

    local ct_args=(-p "$BUILDDIR")
    if [ -f "$ARCH_ROOT/.clang-tidy" ]; then
        ct_args+=(--config-file="$ARCH_ROOT/.clang-tidy")
    fi

    local ct_out
    if ct_out=$(clang-tidy "${ct_args[@]}" "${C_FILES[@]}" 2>&1); then
        pass "clang-tidy: no issues"
    else
        echo "$ct_out"
        fail "clang-tidy: issues found"
        ERRORS=$((ERRORS + 1))
    fi
}

# ─── Header-only checks ──────────────────────────────────────────────────
run_header_checks() {
    header "Header Guard & Include Checks"
    TOOLS_RAN=$((TOOLS_RAN + 1))

    local h_errors=0
    for hf in "${H_FILES[@]}"; do
        local basename
        basename=$(basename "$hf")
        # Project convention: CPU_KERNELS_H (no trailing underscore)
        local guard
        guard=$(echo "$basename" | tr '[:lower:].' '[:upper:]_')

        if ! grep -q "#ifndef $guard" "$hf" 2>/dev/null; then
            fail "$basename: missing or incorrect include guard (expected #ifndef $guard)"
            h_errors=$((h_errors + 1))
        fi
    done

    if [ "$h_errors" -eq 0 ]; then
        pass "All headers have correct include guards"
    else
        ERRORS=$((ERRORS + 1))
    fi
}

# ─── Main ─────────────────────────────────────────────────────────────────
printf "${BOLD}CPU Kernel C Lint Suite${NC}\n"
printf "Arch root: %s\n" "$ARCH_ROOT"
printf "Kernel dir: %s\n" "$KERNEL_DIR"

case "$MODE" in
    --gcc-only)   run_gcc_lint ;;
    --cppcheck)   run_cppcheck ;;
    --clang-tidy) run_clang_tidy ;;
    all)
        run_gcc_lint
        run_cppcheck
        run_clang_tidy
        run_header_checks
        ;;
    *)
        echo "Usage: $0 [--gcc-only|--cppcheck|--clang-tidy|all]"
        exit 2
        ;;
esac

# ─── Summary ──────────────────────────────────────────────────────────────
header "Summary"
if [ "$TOOLS_RAN" -eq 0 ]; then
    fail "No linting tools available. Install gcc, cppcheck, or clang-tidy."
    exit 2
elif [ "$ERRORS" -gt 0 ]; then
    fail "$ERRORS linter(s) reported issues"
    exit 1
else
    pass "All $TOOLS_RAN linter(s) passed"
    exit 0
fi
