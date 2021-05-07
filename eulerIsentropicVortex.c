#include <petsc.h>
#include <petscmath.h>
#include "compressibleFlow.h"
#include "mesh.h"
#include "petscdmplex.h"
#include "petscts.h"

typedef struct {
    PetscInt dim;
    PetscReal rc;
    PetscReal xc;
    PetscReal yc;
    PetscReal L;
    PetscReal Tinf;
    PetscReal rhoInf;
    PetscReal Mac;
    PetscReal MaxInf;
    PetscReal MayInf;
    PetscReal Rgas;
    PetscReal gamma;
} Constants;

typedef struct {
    Constants constants;
    FlowData flowData;
} ProblemSetup;

static PetscErrorCode EulerExact(PetscInt dim, PetscReal time, const PetscReal xyz[], PetscInt Nf, PetscScalar *node, void *ctx) {
    PetscFunctionBeginUser;

    Constants *constants = (Constants *)ctx;

    // define the speed of sound and other variables in the infinity
    PetscReal aInf = PetscSqrtReal(constants->gamma*constants->Rgas*constants->Tinf);
    PetscReal pInf = constants->rhoInf*constants->Rgas*constants->Tinf;
    PetscReal uInf = constants->MaxInf*aInf;
    PetscReal vInf = constants->MayInf*aInf;

    // compute the values for the vortex
    PetscReal Tvort = constants->Tinf/(1.0 + 0.5*(constants->gamma - 1.0)*constants->Mac*constants->Mac);
    PetscReal aVort = PetscSqrtReal(constants->gamma*constants->Rgas*Tvort);

    // compute the values at each location based upon the time and the var field velocity
    PetscReal distTraveledX = time*uInf;
    PetscReal distTraveledY = time*vInf;
    // compute the updated center
    PetscReal xc = constants->xc + distTraveledX;
    PetscReal yc = constants->yc + distTraveledY;

    while(xc > constants->L*2){
        xc -= constants->L*2;
    }
    while(yc > constants->L){
        yc -= constants->L;
    }

    PetscReal x = xyz[0];
    PetscReal y = xyz[1];

    PetscReal xStar = (x- xc)/constants->rc;
    PetscReal yStar = (y - yc)/constants->rc;
    PetscReal rStar = PetscSqrtReal(PetscSqr((x - xc))+ PetscSqr((y - yc)))/constants->rc;

    PetscReal u = uInf - constants->Mac*aVort*yStar* PetscExpReal(0.5*(1.0 - PetscSqr(rStar)));
    PetscReal v = vInf + constants->Mac*aVort*xStar* PetscExpReal(0.5*(1.0 - PetscSqr(rStar)));
    PetscReal machTerm = 0.5*(constants->gamma-1.0)*PetscSqr(constants->Mac);
    PetscReal T = constants->Tinf*(1.0 - machTerm/(1+machTerm)*PetscSqr(rStar)*PetscExpReal(1-PetscSqr(rStar)) );
    PetscReal p = pInf* PetscPowReal(T/constants->Tinf, constants->gamma/(constants->gamma - 1.0));
    PetscReal rho = p/constants->Rgas/T;
    PetscReal e = p/((constants->gamma - 1.0)*rho);
    PetscReal eT = e + 0.5*(u*u + v*v);

    // Store the values
    node[RHO] = rho;
    node[RHOE] = rho*eT;
    node[RHOU + 0] = rho*u;
    node[RHOU + 1] = rho*v;

    PetscFunctionReturn(0);
}

static PetscErrorCode MonitorError(TS ts, PetscInt step, PetscReal time, Vec u, void *ctx) {
    PetscFunctionBeginUser;
    PetscErrorCode     ierr;

    PetscInt interval = 10;
    if(step % interval == 0) {
        // Get the DM
        DM dm;
        ierr = TSGetDM(ts, &dm);
        CHKERRQ(ierr);

        ierr = VecViewFromOptions(u, NULL, "-sol_view");
        CHKERRQ(ierr);

        // Open a vtk viewer
        //    PetscViewer viewer;
        //    char        filename[PETSC_MAX_PATH_LEN];
        //    ierr = PetscSNPrintf(filename,sizeof(filename),"/Users/mcgurn/chrestScratch/results/vortex/flow%.4D.vtu",step);CHKERRQ(ierr);
        //    ierr = PetscViewerVTKOpen(PetscObjectComm((PetscObject)dm),filename,FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
        //    ierr = VecView(u,viewer);CHKERRQ(ierr);
        //    ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);

        ierr = PetscPrintf(PetscObjectComm((PetscObject)dm), "TS at %f\n", time);
        CHKERRQ(ierr);

        // Compute the error
        void *exactCtxs[1];
        PetscErrorCode (*exactFuncs[1])(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar *u, void *ctx);
        PetscDS ds;
        ierr = DMGetDS(dm, &ds);
        CHKERRQ(ierr);

        // Get the exact solution
        ierr = PetscDSGetExactSolution(ds, 0, &exactFuncs[0], &exactCtxs[0]);
        CHKERRQ(ierr);

        // Create an vector to hold the exact solution
        Vec exactVec;
        ierr = VecDuplicate(u, &exactVec);
        CHKERRQ(ierr);
        ierr = DMProjectFunction(dm, time, exactFuncs, exactCtxs, INSERT_ALL_VALUES, exactVec);
        CHKERRQ(ierr);

        ierr = PetscObjectSetName((PetscObject)exactVec, "exact");
        CHKERRQ(ierr);
        ierr = VecViewFromOptions(exactVec, NULL, "-exact_view");
        CHKERRQ(ierr);

        // For each component, compute the l2 norms
        ierr = VecAXPY(exactVec, -1.0, u);
        CHKERRQ(ierr);

        PetscReal ferrors[4];
        ierr = VecSetBlockSize(exactVec, 4);
        CHKERRQ(ierr);

        ierr = PetscPrintf(PETSC_COMM_WORLD, "Timestep: %04d time = %-8.4g \t\n", (int)step, (double)time);
        CHKERRQ(ierr);
        ierr = VecStrideNormAll(exactVec, NORM_2, ferrors);
        CHKERRQ(ierr);
        ierr = PetscPrintf(PETSC_COMM_WORLD, "\tL_2 Error: [%2.3g, %2.3g, %2.3g, %2.3g]\n", (double)(ferrors[0]), (double)(ferrors[1]), (double)(ferrors[2]), (double)(ferrors[3]));
        CHKERRQ(ierr);

        // And the infinity error
        ierr = VecStrideNormAll(exactVec, NORM_INFINITY, ferrors);
        CHKERRQ(ierr);
        ierr = PetscPrintf(PETSC_COMM_WORLD, "\tL_Inf Error: [%2.3g, %2.3g, %2.3g, %2.3g]\n", (double)ferrors[0], (double)ferrors[1], (double)ferrors[2], (double)ferrors[3]);
        CHKERRQ(ierr);

        ierr = PetscObjectSetName((PetscObject)exactVec, "error");
        CHKERRQ(ierr);
        ierr = VecViewFromOptions(exactVec, NULL, "-exact_view");
        CHKERRQ(ierr);
        ierr = VecDestroy(&exactVec);
        CHKERRQ(ierr);
    }
    PetscFunctionReturn(0);
}

static PetscErrorCode PhysicsBoundary_Euler(PetscReal time, const PetscReal *c, const PetscReal *n, const PetscScalar *a_xI, PetscScalar *a_xG, void *ctx) {
    PetscFunctionBeginUser;
    Constants *constants = (Constants *)ctx;

    // Offset the calc assuming the cells are square
    PetscReal x[3];

    for(PetscInt i =0; i < constants->dim; i++){
        x[i] = c[i] + n[i]*0.5;
    }

    EulerExact(constants->dim, time, x, 0, a_xG, ctx);
    PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
    // Setup the problem
    Constants constants;

    // sub sonic
    constants.dim = 2;
    constants.L = 1.0;
    constants.xc = constants.L/2.0;
    constants.yc = constants.L/2.0;
    constants.Tinf =298.0;
    constants.rhoInf = 1.0;
    constants.Rgas = 287.0;
    constants.gamma = 1.4;
    constants.rc = .1*constants.L;
    constants.gamma = 1.4;
    constants.Mac = 0.3;
    constants.MaxInf = 0.3;
    constants.MayInf = 0.0;

    PetscErrorCode ierr;
    // create the mesh
    // setup the ts
    DM dm;                 /* problem definition */
    TS ts;                 /* timestepper */

    // initialize petsc and mpi
    PetscInitialize(&argc, &argv, NULL, "HELP");

    // Create a ts
    ierr = TSCreate(PETSC_COMM_WORLD, &ts);CHKERRQ(ierr);
    ierr = TSSetProblemType(ts, TS_NONLINEAR);CHKERRQ(ierr);
    ierr = TSSetType(ts, TSRK);CHKERRQ(ierr);
    ierr = TSSetExactFinalTime(ts, TS_EXACTFINALTIME_MATCHSTEP);CHKERRQ(ierr);

    PetscInt lengthFactor = 2;
    ierr =  PetscOptionsGetInt(NULL, NULL, "lengthFactor", &lengthFactor, NULL);CHKERRQ(ierr);
    PetscPrintf(PETSC_COMM_WORLD, "LengthFactor %d\n", lengthFactor);

    // Create a mesh
    // hard code the problem setup
    PetscReal start[] = {0.0, 0.0};
    PetscReal end[] = {constants.L*lengthFactor, constants.L};
    PetscReal nxHeight = 10;
    PetscInt nx[] = {lengthFactor*nxHeight, nxHeight};
    ierr = DMPlexCreateBoxMesh(PETSC_COMM_WORLD, constants.dim, PETSC_FALSE, nx, start, end, NULL, PETSC_TRUE, &dm);CHKERRQ(ierr);

    // Setup the flow data
    FlowData flowData;     /* store some of the flow data*/
    ierr = FlowCreate(&flowData);CHKERRQ(ierr);

    // Combine the flow data
    ProblemSetup problemSetup;
    problemSetup.flowData = flowData;
    problemSetup.constants = constants;

    //Setup
    CompressibleFlow_SetupDiscretization(flowData, &dm);

    // Add in the flow parameters
    PetscScalar params[TOTAL_COMPRESSIBLE_FLOW_PARAMETERS];
    params[CFL] = 0.5;
    params[GAMMA] = constants.gamma;

    // set up the finite volume fluxes
    CompressibleFlow_StartProblemSetup(flowData, TOTAL_COMPRESSIBLE_FLOW_PARAMETERS, params);
    DMView(flowData->dm, PETSC_VIEWER_STDOUT_WORLD);
    // Add in any boundary conditions
    PetscDS prob;
    ierr = DMGetDS(flowData->dm, &prob);
    CHKERRABORT(PETSC_COMM_WORLD, ierr);
    const PetscInt idsLeft[]= {1, 2, 3, 4};
    ierr = PetscDSAddBoundary(prob, DM_BC_NATURAL_RIEMANN, "wall left", "Face Sets", 0, 0, NULL, (void (*)(void))PhysicsBoundary_Euler, NULL, 4, idsLeft, &constants);CHKERRQ(ierr);

    // Complete the problem setup
    ierr = CompressibleFlow_CompleteProblemSetup(flowData, ts);
    CHKERRABORT(PETSC_COMM_WORLD, ierr);

    // Name the flow field
    ierr = PetscObjectSetName(((PetscObject)flowData->flowField), "Numerical Solution");
    CHKERRABORT(PETSC_COMM_WORLD, ierr);

    // Setup the TS
    ierr = TSSetFromOptions(ts);
    CHKERRABORT(PETSC_COMM_WORLD, ierr);
    ierr = TSMonitorSet(ts, MonitorError, &constants, NULL);CHKERRQ(ierr);
    CHKERRABORT(PETSC_COMM_WORLD, ierr);

    // set the initial conditions
    PetscErrorCode     (*func[2]) (PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar *u, void *ctx) = {EulerExact};
    void* ctxs[1] ={&constants};
    ierr    = DMProjectFunction(flowData->dm,0.0,func,ctxs,INSERT_ALL_VALUES,flowData->flowField);CHKERRQ(ierr);

    // for the mms, add the exact solution
    ierr = PetscDSSetExactSolution(prob, 0, EulerExact, &constants);CHKERRQ(ierr);

    // Output the mesh
    ierr = DMViewFromOptions(flowData->dm, NULL, "-dm_view");CHKERRQ(ierr);

    // Compute the end time so it goes around once
    PetscReal aInf = PetscSqrtReal(constants.gamma*constants.Rgas*constants.Tinf);
    PetscReal u_x = constants.MaxInf*aInf;
    PetscReal endTime = constants.L/u_x;

    TSSetMaxTime(ts, endTime);
    TSSetMaxSteps(ts, 2000);
    PetscDSView(prob, PETSC_VIEWER_STDOUT_WORLD);
    PetscLogStage solveStage;
    PetscLogStageRegister("TSSolve",&solveStage);

    PetscLogStagePush(solveStage);
    ierr = TSSolve(ts,flowData->flowField);CHKERRQ(ierr);
    PetscLogStagePop();

    FlowDestroy(&flowData);
    TSDestroy(&ts);

    return PetscFinalize();

}
