static char help[] = "Example ablateCore Client using  incompressible flow";

/** Example Argumetns
 -dm_plex_separate_marker -dm_refine 0 \
 -vel_petscspace_degree 2 -pres_petscspace_degree 1 -temp_petscspace_degree 1 \
 -dmts_check .001 -ts_max_steps 4 -ts_dt 0.1 \
 -ksp_type fgmres -ksp_gmres_restart 10 -ksp_rtol 1.0e-9 -ksp_error_if_not_converged \
 -pc_type fieldsplit -pc_fieldsplit_0_fields 0,2 -pc_fieldsplit_1_fields 1 -pc_fieldsplit_type schur -pc_fieldsplit_schur_factorization_type full \
 -fieldsplit_0_pc_type lu \
 -fieldsplit_pressure_ksp_rtol 1e-10 -fieldsplit_pressure_pc_type jacobi
 */

#include <petsc.h>
#include "flow.h"
#include "incompressibleFlow.h"
#include "mesh.h"

typedef PetscErrorCode (*ExactFunction)(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar *u, void *ctx);

typedef void (*IntegrandTestFunction)(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt *uOff, const PetscInt *uOff_x, const PetscScalar *u, const PetscScalar *u_t, const PetscScalar *u_x,
                                      const PetscInt *aOff, const PetscInt *aOff_x, const PetscScalar *a, const PetscScalar *a_t, const PetscScalar *a_x, PetscReal t, const PetscReal *X,
                                      PetscInt numConstants, const PetscScalar *constants, PetscScalar *f0);

#define SourceFunction(FUNC)            \
    FUNC(PetscInt dim,                  \
         PetscInt Nf,                   \
         PetscInt NfAux,                \
         const PetscInt uOff[],         \
         const PetscInt uOff_x[],       \
         const PetscScalar u[],         \
         const PetscScalar u_t[],       \
         const PetscScalar u_x[],       \
         const PetscInt aOff[],         \
         const PetscInt aOff_x[],       \
         const PetscScalar a[],         \
         const PetscScalar a_t[],       \
         const PetscScalar a_x[],       \
         PetscReal t,                   \
         const PetscReal X[],           \
         PetscInt numConstants,         \
         const PetscScalar constants[], \
         PetscScalar f0[])

// store the pointer to the provided test function from the solver
static IntegrandTestFunction f0_v_original;
static IntegrandTestFunction f0_w_original;
static IntegrandTestFunction f0_q_original;

static PetscErrorCode SetInitialConditions(TS ts, Vec u) {
    DM dm;
    PetscReal t;
    PetscErrorCode ierr;

    PetscFunctionBegin;
    ierr = TSGetDM(ts, &dm);
    CHKERRQ(ierr);
    ierr = TSGetTime(ts, &t);
    CHKERRQ(ierr);

    // This function Tags the u vector as the exact solution.  We need to copy the values to prevent this.
    Vec e;
    ierr = VecDuplicate(u, &e);
    CHKERRQ(ierr);
    ierr = DMComputeExactSolution(dm, t, e, NULL);
    CHKERRQ(ierr);
    ierr = VecCopy(e, u);
    CHKERRQ(ierr);
    ierr = VecDestroy(&e);
    CHKERRQ(ierr);

    // get the flow to apply the completeFlowInitialization method
    ierr = IncompressibleFlow_CompleteFlowInitialization(dm, u);
    CHKERRQ(ierr);

    PetscFunctionReturn(0);
}

static PetscErrorCode MonitorError(TS ts, PetscInt step, PetscReal crtime, Vec u, void *ctx) {
    PetscErrorCode (*exactFuncs[3])(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar *u, void *ctx);
    void *ctxs[3];
    DM dm;
    PetscDS ds;
    Vec v;
    PetscReal ferrors[3];
    PetscInt f;
    PetscErrorCode ierr;

    PetscFunctionBeginUser;
    ierr = TSGetDM(ts, &dm);
    CHKERRQ(ierr);
    ierr = DMGetDS(dm, &ds);
    CHKERRQ(ierr);

    ierr = VecViewFromOptions(u, NULL, "-vec_view_monitor");
    CHKERRABORT(PETSC_COMM_WORLD, ierr);

    for (f = 0; f < 3; ++f) {
        ierr = PetscDSGetExactSolution(ds, f, &exactFuncs[f], &ctxs[f]);
        CHKERRABORT(PETSC_COMM_WORLD, ierr);
    }
    ierr = DMComputeL2FieldDiff(dm, crtime, exactFuncs, ctxs, u, ferrors);
    CHKERRABORT(PETSC_COMM_WORLD, ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD, "Timestep: %04d time = %-8.4g \t L_2 Error: [%2.3g, %2.3g, %2.3g]\n", (int)step, (double)crtime, (double)ferrors[0], (double)ferrors[1], (double)ferrors[2]);
    CHKERRABORT(PETSC_COMM_WORLD, ierr);

    ierr = DMGetGlobalVector(dm, &u);
    CHKERRABORT(PETSC_COMM_WORLD, ierr);
    // ierr = TSGetSolution(ts, &u);CHKERRABORT(PETSC_COMM_WORLD, ierr);
    //    ierr = PetscObjectSetName((PetscObject)u, "Numerical Solution");
    CHKERRABORT(PETSC_COMM_WORLD, ierr);
    CHKERRABORT(PETSC_COMM_WORLD, ierr);
    ierr = DMRestoreGlobalVector(dm, &u);
    CHKERRABORT(PETSC_COMM_WORLD, ierr);

    ierr = DMGetGlobalVector(dm, &v);
    CHKERRABORT(PETSC_COMM_WORLD, ierr);
    // ierr = VecSet(v, 0.0);CHKERRABORT(PETSC_COMM_WORLD, ierr);
    ierr = DMProjectFunction(dm, 0.0, exactFuncs, ctxs, INSERT_ALL_VALUES, v);
    CHKERRABORT(PETSC_COMM_WORLD, ierr);
    ierr = PetscObjectSetName((PetscObject)v, "Exact Solution");
    CHKERRABORT(PETSC_COMM_WORLD, ierr);
    ierr = VecViewFromOptions(v, NULL, "-exact_vec_view");
    CHKERRABORT(PETSC_COMM_WORLD, ierr);
    ierr = DMRestoreGlobalVector(dm, &v);
    CHKERRABORT(PETSC_COMM_WORLD, ierr);

    PetscFunctionReturn(0);
}

// helper functions for generated code
static PetscReal Power(PetscReal x, PetscInt exp) { return PetscPowReal(x, exp); }
static PetscReal Cos(PetscReal x) { return PetscCosReal(x); }
static PetscReal Sin(PetscReal x) { return PetscSinReal(x); }

/*
  CASE: incompressible quadratic
  In 2D we use exact solution:

    u = t + x^2 + y^2
    v = t + 2x^2 - 2xy
    p = x + y - 1
    T = t + x + y
  so that

    \nabla \cdot u = 2x - 2x = 0

  see docs/content/formulations/incompressibleFlow/solutions/Incompressible_2D_Quadratic_MMS.nb
*/
static PetscErrorCode incompressible_quadratic_u(PetscInt Dim, PetscReal time, const PetscReal *X, PetscInt Nf, PetscScalar *u, void *ctx) {
    u[0] = time + X[0] * X[0] + X[1] * X[1];
    u[1] = time + 2.0 * X[0] * X[0] - 2.0 * X[0] * X[1];
    return 0;
}
static PetscErrorCode incompressible_quadratic_u_t(PetscInt Dim, PetscReal time, const PetscReal *X, PetscInt Nf, PetscScalar *u, void *ctx) {
    u[0] = 1.0;
    u[1] = 1.0;
    return 0;
}

static PetscErrorCode incompressible_quadratic_p(PetscInt Dim, PetscReal time, const PetscReal *X, PetscInt Nf, PetscScalar *p, void *ctx) {
    p[0] = X[0] + X[1] - 1.0;
    return 0;
}

static PetscErrorCode incompressible_quadratic_T(PetscInt Dim, PetscReal time, const PetscReal *X, PetscInt Nf, PetscScalar *T, void *ctx) {
    T[0] = time + X[0] + X[1];
    return 0;
}
static PetscErrorCode incompressible_quadratic_T_t(PetscInt Dim, PetscReal time, const PetscReal *X, PetscInt Nf, PetscScalar *T, void *ctx) {
    T[0] = 1.0;
    return 0;
}

/* f0_v = du/dt - f */
static void SourceFunction(f0_incompressible_quadratic_v) {
    f0_v_original(dim, Nf, NfAux, uOff, uOff_x, u, u_t, u_x, aOff, aOff_x, a, a_t, a_x, t, X, numConstants, constants, f0);
    const PetscReal rho = 1.0;
    const PetscReal S = constants[STROUHAL];
    const PetscReal mu = constants[MU];
    const PetscReal R = constants[REYNOLDS];
    const PetscReal x = X[0];
    const PetscReal y = X[1];

    f0[0] -= 1 - (4. * mu) / R + rho * S + 2 * rho * y * (t + 2 * Power(x, 2) - 2 * x * y) + 2 * rho * x * (t + Power(x, 2) + Power(y, 2));
    f0[1] -= 1 - (4. * mu) / R + rho * S - 2 * rho * x * (t + 2 * Power(x, 2) - 2 * x * y) + rho * (4 * x - 2 * y) * (t + Power(x, 2) + Power(y, 2));
}

/* f0_w = dT/dt + u.grad(T) - Q */
static void SourceFunction(f0_incompressible_quadratic_w) {
    f0_w_original(dim, Nf, NfAux, uOff, uOff_x, u, u_t, u_x, aOff, aOff_x, a, a_t, a_x, t, X, numConstants, constants, f0);

    const PetscReal rho = 1.0;
    const PetscReal S = constants[STROUHAL];
    const PetscReal Cp = constants[CP];
    const PetscReal x = X[0];
    const PetscReal y = X[1];

    f0[0] -= Cp * rho * (S + 2 * t + 3 * Power(x, 2) - 2 * x * y + Power(y, 2));
}

int main(int argc, char *argv[]) {
    DM dm;                 /* problem definition */
    TS ts;                 /* timestepper */
    PetscBag parameterBag; /* constant flow parameters */
    FlowData flowData;     /* store some of the flow data*/
    PetscReal t;
    PetscErrorCode ierr;

    // initialize petsc and mpi
    PetscInitialize(&argc, &argv, NULL, help);

    // setup the ts
    ierr = TSCreate(PETSC_COMM_WORLD, &ts);
    CHKERRABORT(PETSC_COMM_WORLD, ierr);
    ierr = CreateMesh(PETSC_COMM_WORLD, &dm, PETSC_TRUE, 2);
    CHKERRABORT(PETSC_COMM_WORLD, ierr);
    ierr = TSSetDM(ts, dm);
    CHKERRABORT(PETSC_COMM_WORLD, ierr);
    ierr = TSSetExactFinalTime(ts, TS_EXACTFINALTIME_MATCHSTEP);
    CHKERRABORT(PETSC_COMM_WORLD, ierr);

    // Setup the flow data
    ierr = FlowCreate(&flowData);
    CHKERRABORT(PETSC_COMM_WORLD, ierr);

    // setup problem
    ierr = IncompressibleFlow_SetupDiscretization(flowData, dm);
    CHKERRABORT(PETSC_COMM_WORLD, ierr);

    // get the flow parameters from options
    IncompressibleFlowParameters *flowParameters;
    ierr = IncompressibleFlow_ParametersFromPETScOptions(&parameterBag);
    CHKERRABORT(PETSC_COMM_WORLD, ierr);
    ierr = PetscBagGetData(parameterBag, (void **)&flowParameters);
    CHKERRABORT(PETSC_COMM_WORLD, ierr);

    // Start the problem setup
    PetscScalar constants[TOTAL_INCOMPRESSIBLE_FLOW_PARAMETERS];
    ierr = IncompressibleFlow_PackParameters(flowParameters, constants);
    CHKERRABORT(PETSC_COMM_WORLD, ierr);
    ierr = IncompressibleFlow_StartProblemSetup(flowData, TOTAL_INCOMPRESSIBLE_FLOW_PARAMETERS, constants);
    CHKERRABORT(PETSC_COMM_WORLD, ierr);

    // Override problem with source terms, boundary, and set the exact solution
    {
        PetscDS prob;
        ierr = DMGetDS(dm, &prob);
        CHKERRABORT(PETSC_COMM_WORLD, ierr);

        // V, W Test Function
        IntegrandTestFunction tempFunctionPointer;
        ierr = PetscDSGetResidual(prob, VTEST, &f0_v_original, &tempFunctionPointer);
        CHKERRABORT(PETSC_COMM_WORLD, ierr);
        ierr = PetscDSSetResidual(prob, VTEST, f0_incompressible_quadratic_v, tempFunctionPointer);
        CHKERRABORT(PETSC_COMM_WORLD, ierr);

        ierr = PetscDSGetResidual(prob, WTEST, &f0_w_original, &tempFunctionPointer);
        CHKERRABORT(PETSC_COMM_WORLD, ierr);
        ierr = PetscDSSetResidual(prob, WTEST, f0_incompressible_quadratic_w, tempFunctionPointer);
        CHKERRABORT(PETSC_COMM_WORLD, ierr);

        /* Setup Boundary Conditions */
        PetscInt id;
        id = 3;
        ierr = PetscDSAddBoundary(prob, DM_BC_ESSENTIAL, "top wall velocity", "marker", VEL, 0, NULL, (void (*)(void))incompressible_quadratic_u, (void (*)(void))incompressible_quadratic_u_t, 1, &id, parameterBag);
        CHKERRABORT(PETSC_COMM_WORLD, ierr);
        id = 1;
        ierr =
            PetscDSAddBoundary(prob, DM_BC_ESSENTIAL, "bottom wall velocity", "marker", VEL, 0, NULL, (void (*)(void))incompressible_quadratic_u, (void (*)(void))incompressible_quadratic_u_t, 1, &id, parameterBag);
        CHKERRABORT(PETSC_COMM_WORLD, ierr);
        id = 2;
        ierr =
            PetscDSAddBoundary(prob, DM_BC_ESSENTIAL, "right wall velocity", "marker", VEL, 0, NULL, (void (*)(void))incompressible_quadratic_u, (void (*)(void))incompressible_quadratic_u_t, 1, &id, parameterBag);
        CHKERRABORT(PETSC_COMM_WORLD, ierr);
        id = 4;
        ierr =
            PetscDSAddBoundary(prob, DM_BC_ESSENTIAL, "left wall velocity", "marker", VEL, 0, NULL, (void (*)(void))incompressible_quadratic_u, (void (*)(void))incompressible_quadratic_u_t, 1, &id, parameterBag);
        CHKERRABORT(PETSC_COMM_WORLD, ierr);
        id = 3;
        ierr = PetscDSAddBoundary(prob, DM_BC_ESSENTIAL, "top wall temp", "marker", TEMP, 0, NULL, (void (*)(void))incompressible_quadratic_T, (void (*)(void))incompressible_quadratic_T_t, 1, &id, parameterBag);
        CHKERRABORT(PETSC_COMM_WORLD, ierr);
        id = 1;
        ierr = PetscDSAddBoundary(prob, DM_BC_ESSENTIAL, "bottom wall temp", "marker", TEMP, 0, NULL, (void (*)(void))incompressible_quadratic_T, (void (*)(void))incompressible_quadratic_T_t, 1, &id, parameterBag);
        CHKERRABORT(PETSC_COMM_WORLD, ierr);
        id = 2;
        ierr = PetscDSAddBoundary(prob, DM_BC_ESSENTIAL, "right wall temp", "marker", TEMP, 0, NULL, (void (*)(void))incompressible_quadratic_T, (void (*)(void))incompressible_quadratic_T_t, 1, &id, parameterBag);
        CHKERRABORT(PETSC_COMM_WORLD, ierr);
        id = 4;
        ierr = PetscDSAddBoundary(prob, DM_BC_ESSENTIAL, "left wall temp", "marker", TEMP, 0, NULL, (void (*)(void))incompressible_quadratic_T, (void (*)(void))incompressible_quadratic_T_t, 1, &id, parameterBag);
        CHKERRABORT(PETSC_COMM_WORLD, ierr);

        // Set the exact solution
        ierr = PetscDSSetExactSolution(prob, VEL, incompressible_quadratic_u, parameterBag);
        CHKERRABORT(PETSC_COMM_WORLD, ierr);
        ierr = PetscDSSetExactSolution(prob, PRES, incompressible_quadratic_p, parameterBag);
        CHKERRABORT(PETSC_COMM_WORLD, ierr);
        ierr = PetscDSSetExactSolution(prob, TEMP, incompressible_quadratic_T, parameterBag);
        CHKERRABORT(PETSC_COMM_WORLD, ierr);
        ierr = PetscDSSetExactSolutionTimeDerivative(prob, VEL, incompressible_quadratic_u_t, parameterBag);
        CHKERRABORT(PETSC_COMM_WORLD, ierr);
        ierr = PetscDSSetExactSolutionTimeDerivative(prob, PRES, NULL, parameterBag);
        CHKERRABORT(PETSC_COMM_WORLD, ierr);
        ierr = PetscDSSetExactSolutionTimeDerivative(prob, TEMP, incompressible_quadratic_T_t, parameterBag);
        CHKERRABORT(PETSC_COMM_WORLD, ierr);
    }
    ierr = IncompressibleFlow_CompleteProblemSetup(flowData, ts);
    CHKERRABORT(PETSC_COMM_WORLD, ierr);

    // Name the flow field
    ierr = PetscObjectSetName(((PetscObject)flowData->flowField), "Numerical Solution");
    CHKERRABORT(PETSC_COMM_WORLD, ierr);

    // Setup the TS
    ierr = TSSetFromOptions(ts);
    CHKERRABORT(PETSC_COMM_WORLD, ierr);

    // Set initial conditions from the exact solution
    ierr = TSSetComputeInitialCondition(ts, SetInitialConditions);
    CHKERRABORT(PETSC_COMM_WORLD, ierr); /* Must come after SetFromOptions() */
    ierr = SetInitialConditions(ts, flowData->flowField);

    CHKERRABORT(PETSC_COMM_WORLD, ierr);
    ierr = TSGetTime(ts, &t);
    CHKERRABORT(PETSC_COMM_WORLD, ierr);
    ierr = DMSetOutputSequenceNumber(dm, 0, t);
    CHKERRABORT(PETSC_COMM_WORLD, ierr);
    ierr = DMTSCheckFromOptions(ts, flowData->flowField);
    CHKERRABORT(PETSC_COMM_WORLD, ierr);
    ierr = TSMonitorSet(ts, MonitorError, NULL, NULL);
    CHKERRABORT(PETSC_COMM_WORLD, ierr);

    ierr = TSSolve(ts, flowData->flowField);
    CHKERRABORT(PETSC_COMM_WORLD, ierr);

    // Compare the actual vs expected values
    ierr = DMTSCheckFromOptions(ts, flowData->flowField);
    CHKERRABORT(PETSC_COMM_WORLD, ierr);

    // Cleanup
    ierr = FlowDestroy(&flowData);
    CHKERRABORT(PETSC_COMM_WORLD, ierr);
    ierr = DMDestroy(&dm);
    CHKERRABORT(PETSC_COMM_WORLD, ierr);
    ierr = TSDestroy(&ts);
    CHKERRABORT(PETSC_COMM_WORLD, ierr);
    ierr = PetscBagDestroy(&parameterBag);
    CHKERRABORT(PETSC_COMM_WORLD, ierr);
    return PetscFinalize();
}