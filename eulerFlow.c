#include <petsc.h>
#include "compressibleFlow.h"
#include "mesh.h"

#define DIM 2                   /* Geometric dimension */


typedef struct {
    InitialConditions initialConditions;
    StarState starState;
} ProblemSetup;


static PetscErrorCode SetExactSolutionRho(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar *u, void *ctx){
    ProblemSetup* prob = (ProblemSetup*)ctx;

    PetscReal xDt = (x[0]-prob->initialConditions.length/2)/time;
    EulerNode uu;
    SetExactSolutionAtPoint(dim, xDt, &prob->initialConditions, &prob->starState, &uu);

    u[0] = uu.rho;
    return 0;
}

static PetscErrorCode SetExactSolutionRhoU(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar *u, void *ctx){
    ProblemSetup* prob = (ProblemSetup*)ctx;

    PetscReal xDt = (x[0]-prob->initialConditions.length/2)/time;
    EulerNode uu;
    SetExactSolutionAtPoint(dim, xDt, &prob->initialConditions, &prob->starState, &uu);
    u[0] = uu.rhoU[0];
    u[1] = uu.rhoU[1];
    return 0;
}

static PetscErrorCode SetExactSolutionRhoE(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar *u, void *ctx){
    ProblemSetup* prob = (ProblemSetup*)ctx;

    PetscReal xDt = (x[0]-prob->initialConditions.length/2)/time;
    EulerNode uu;
    SetExactSolutionAtPoint(dim, xDt, &prob->initialConditions, &prob->starState, &uu);
    u[0] = uu.rhoE;
    return 0;
}

static PetscErrorCode PrintVector(DM dm, Vec v, PetscInt step, const char * fileName){
    Vec                cellgeom;
    PetscErrorCode ierr = DMPlexGetGeometryFVM(dm, NULL, &cellgeom, NULL);CHKERRQ(ierr);
    PetscInt cStart, cEnd;
        ierr = DMPlexGetSimplexOrBoxCells(dm, 0, &cStart, &cEnd);CHKERRQ(ierr);
    DM dmCell;
    ierr = VecGetDM(cellgeom, &dmCell);CHKERRQ(ierr);
    const PetscScalar *cgeom;
    ierr = VecGetArrayRead(cellgeom, &cgeom);CHKERRQ(ierr);
    const PetscScalar      *x;
    ierr = VecGetArrayRead(v, &x);CHKERRQ(ierr);
    // print the header for each file
    char filename[100];
    ierr = PetscSNPrintf(filename,sizeof(filename),"%s.%d.txt",fileName, step);CHKERRQ(ierr);

    PetscInt rank = 0;
    PetscInt size;
    ierr = MPI_Comm_rank(PetscObjectComm(dm), &rank);CHKERRMPI(ierr);
    ierr = MPI_Comm_size(PetscObjectComm(dm), &size);CHKERRMPI(ierr);

    for(PetscInt r =0; r < size; r++ ) {
        if(r == rank) {
            FILE *fptr;
            if(r == 0){
                fptr = fopen(filename, "w");
                fprintf(fptr, "x rho u e\n");
            }else{
                fptr = fopen(filename, "a");
            }
            for (PetscInt c = cStart; c < cEnd; ++c) {
                PetscFVCellGeom *cg;
                const EulerNode *xc;

                ierr = DMPlexPointLocalRead(dmCell, c, cgeom, &cg);
                CHKERRQ(ierr);
                ierr = DMPlexPointGlobalFieldRead(dm, c, 0, x, &xc);
                CHKERRQ(ierr);
                if(xc) {// must be real cell and not ghost
                    PetscReal u0 = xc->rhoU[0] / xc->rho;
                    fprintf(fptr, "%f %f %f %f\n", cg->centroid[0], xc->rho, u0, (xc->rhoE / xc->rho) - 0.5 * u0 * u0);
                }
            }

            fclose(fptr);
        }
        MPI_Barrier(PetscObjectComm(dm));
    }
    ierr = VecRestoreArrayRead(cellgeom, &cgeom);CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(v, &x);CHKERRQ(ierr);
    return 0;
}

static PetscErrorCode MonitorError(TS ts, PetscInt step, PetscReal time, Vec u, void *ctx) {
    PetscFunctionBeginUser;
    PetscErrorCode     ierr;

    // Get the DM
    DM dm;
    ierr = TSGetDM(ts, &dm);CHKERRQ(ierr);

    // Create a copy of the u vector
    Vec e;
    ierr = DMCreateGlobalVector(dm, &e);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject)e, "exact");CHKERRQ(ierr);

    // Set the values
    PetscErrorCode     (*func[3]) (PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar *u, void *ctx) = {{SetExactSolutionRho, SetExactSolutionRhoU, SetExactSolutionRhoE}};
    void* ctxs[3] ={ctx, ctx, ctx};
    ierr    = DMProjectFunction(dm,time,func,ctxs,INSERT_ALL_VALUES,e);CHKERRQ(ierr);

    // just print to a file for now
    ierr = PrintVector(dm, e, step, "exact");CHKERRQ(ierr);
    ierr = PrintVector(dm, u, step, "solution");CHKERRQ(ierr);

    PetscPrintf(PETSC_COMM_WORLD, "TS %d: %f\n", step, time);

    DMRestoreGlobalVector(dm, &e);
    PetscFunctionReturn(0);
}

static PetscErrorCode PhysicsBoundary_Euler_Mirror(PetscReal time, const PetscReal *c, const PetscReal *n, const PetscScalar *a_xI, PetscScalar *a_xG, void *ctx)
{
    const EulerNode *xI = (const EulerNode*)a_xI;
    EulerNode       *xG = (EulerNode*)a_xG;
    ProblemSetup* prob = (ProblemSetup*)ctx;
    PetscFunctionBeginUser;
    xG->rho = xI->rho;
    xG->rhoE = xI->rhoE;
    xG->rhoU[0] = xI->rhoU[0];
    xG->rhoU[1] = xI->rhoU[1];

    PetscFunctionReturn(0);
}

/* PetscReal* => EulerNode* conversion */
static PetscErrorCode PhysicsBoundary_Euler_Left(PetscReal time, const PetscReal *c, const PetscReal *n, const PetscScalar *a_xI, PetscScalar *a_xG, void *ctx)
{
    const EulerNode *xI = (const EulerNode*)a_xI;
    EulerNode       *xG = (EulerNode*)a_xG;
    ProblemSetup* prob = (ProblemSetup*)ctx;
    PetscFunctionBeginUser;
    xG->rho = prob->initialConditions.rhoL;
    PetscReal eT = prob->initialConditions.rhoL*((prob->initialConditions.pL /(prob->initialConditions.gamma -1) / prob->initialConditions.rhoL) + 0.5 * prob->initialConditions.uL * prob->initialConditions.uL);
    xG->rhoE = eT;
    xG->rhoU[0] = prob->initialConditions.rhoL * prob->initialConditions.uL;
    xG->rhoU[1] = 0.0;

    PetscFunctionReturn(0);
}

/* PetscReal* => EulerNode* conversion */
static PetscErrorCode PhysicsBoundary_Euler_Right(PetscReal time, const PetscReal *c, const PetscReal *n, const PetscScalar *a_xI, PetscScalar *a_xG, void *ctx)
{
    const EulerNode *xI = (const EulerNode*)a_xI;
    EulerNode       *xG = (EulerNode*)a_xG;
    ProblemSetup*        prob = (ProblemSetup*)ctx;
    PetscFunctionBeginUser;
    xG->rho = prob->initialConditions.rhoR;
    PetscReal eT = prob->initialConditions.rhoR*((prob->initialConditions.pR /(prob->initialConditions.gamma -1)/ prob->initialConditions.rhoR) + 0.5 * prob->initialConditions.uR * prob->initialConditions.uR);
    xG->rhoE = eT;
    xG->rhoU[0] = prob->initialConditions.rhoR * prob->initialConditions.uR;
    xG->rhoU[1] = 0.0;

    PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
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
    ierr = TSSetType(ts, TSEULER);CHKERRQ(ierr);
    ierr = TSSetExactFinalTime(ts, TS_EXACTFINALTIME_MATCHSTEP);CHKERRQ(ierr);

    // Create a mesh
    // hard code the problem setup
    PetscReal start[] = {0.0, 0.0};
    PetscReal end[] = {1.0, 1};
    PetscInt nx[] = {100, 1};
    DMBoundaryType bcType[] = {DM_BOUNDARY_NONE, DM_BOUNDARY_NONE};
    ierr = DMPlexCreateBoxMesh(PETSC_COMM_WORLD, DIM, PETSC_FALSE, nx, start, end, bcType, PETSC_TRUE, &dm);CHKERRQ(ierr);

// Setup the problem
    ProblemSetup problem;

    // case 1 - Sod problem
    problem.initialConditions.rhoL=1.0;
    problem.initialConditions.uL=0.0;
    problem.initialConditions.pL=1.0;
    problem.initialConditions.rhoR=0.125;
    problem.initialConditions.uR=0.0;
    problem.initialConditions.pR=0.1;
    problem.initialConditions.maxTime = 1.0;
    problem.initialConditions.length = 1;
    problem.initialConditions.gamma = 1.4;

    // Compute the star state
    ierr = DetermineStarState(&problem.initialConditions, &problem.starState);CHKERRQ(ierr);

    // Setup the flow data
    FlowData flowData;     /* store some of the flow data*/
    ierr = FlowCreate(&flowData);CHKERRQ(ierr);

    // Add in the flow parameters
    EulerFlowParameters flowParam;
    flowParam.cfl = .5;
    flowParam.gamma = problem.initialConditions.gamma;

    //Store the flow params
    CompressibleFlow_SetupFlowParameters(flowData, &flowParam);

    //Setup
    CompressibleFlow_SetupDiscretization(flowData, dm);

    // set up the finite volume fluxes
    CompressibleFlow_StartProblemSetup(flowData);

    // Add in any boundary conditions
    PetscDS prob;
    ierr = DMGetDS(flowData->dm, &prob);
    CHKERRABORT(PETSC_COMM_WORLD, ierr);
    const PetscInt idsLeft[]= {4};
    ierr = PetscDSAddBoundary(prob, DM_BC_NATURAL_RIEMANN, "wall left", "Face Sets", 0, 0, NULL, (void (*)(void)) PhysicsBoundary_Euler_Left, NULL, 1, idsLeft, &problem);CHKERRQ(ierr);
    const PetscInt idsRight[]= {2};
    ierr = PetscDSAddBoundary(prob, DM_BC_NATURAL_RIEMANN, "wall right", "Face Sets", 0, 0, NULL, (void (*)(void)) PhysicsBoundary_Euler_Right, NULL, 1, idsRight, &problem);CHKERRQ(ierr);
    const PetscInt mirror[]= {1, 3};
    ierr = PetscDSAddBoundary(prob, DM_BC_NATURAL_RIEMANN, "mirrorWall", "Face Sets", 0, 0, NULL, (void (*)(void)) PhysicsBoundary_Euler_Mirror, NULL, 2, mirror, &problem);CHKERRQ(ierr);

    // Complete the problem setup
    ierr = CompressibleFlow_CompleteProblemSetup(flowData, ts);

    // Name the flow field
    ierr = PetscObjectSetName(((PetscObject)flowData->flowField), "Numerical Solution");
    CHKERRABORT(PETSC_COMM_WORLD, ierr);

    // Setup the TS
    ierr = TSSetFromOptions(ts);
    CHKERRABORT(PETSC_COMM_WORLD, ierr);
    ierr = TSMonitorSet(ts, MonitorError, &problem, NULL);CHKERRQ(ierr);
    CHKERRABORT(PETSC_COMM_WORLD, ierr);
    ierr = TSSetMaxTime(ts,problem.initialConditions.maxTime);CHKERRQ(ierr);

    // set the initial conditions
    PetscErrorCode     (*func[3]) (PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar *u, void *ctx) = {SetExactSolutionRho, SetExactSolutionRhoU, SetExactSolutionRhoE};
    void* ctxs[3] ={&problem, &problem, &problem};
    ierr    = DMProjectFunction(flowData->dm,0.0,func,ctxs,INSERT_ALL_VALUES,flowData->flowField);CHKERRQ(ierr);

    ierr = TSSolve(ts,flowData->flowField);CHKERRQ(ierr);


    return PetscFinalize();

}