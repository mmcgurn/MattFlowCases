#include <petsc.h>
#include "compressibleFlow.h"
#include "mesh.h"

#define DIM 2                   /* Geometric dimension */

typedef struct {
    PetscReal gamma;
    PetscReal rhoL;
    PetscReal rhoR;
    PetscReal uL;
    PetscReal uR;
    PetscReal pL;
    PetscReal pR;
    PetscReal maxTime;
    PetscReal length;
} InitialConditions;

typedef struct {
    PetscReal rho;
    PetscReal rhoU[2];//Hardcode for 2 for now
    PetscReal rhoE;
} EulerNode;

typedef struct {
    PetscReal pstar,ustar,rhostarL,astarL,SL,SHL,STL,rhostarR,astarR,SR,SHR,STR,gamm1,gamp1;
} StarState;



typedef struct {
    InitialConditions initialConditions;
    StarState starState;
} ProblemSetup;


typedef struct{
    PetscReal f;
    PetscReal fprm;
} PressureFunction;

static PressureFunction f_and_fprm_rarefaction(PetscReal pstar, PetscReal pLR, PetscReal aLR, PetscReal gam, PetscReal gamm1,PetscReal gamp1) {
    // compute value of pressure function for rarefaction
    PressureFunction function;
    function.f = ((2. * aLR) / gamm1) * (pow(pstar / pLR, 0.5 * gamm1 / gam) - 1.);
    function.fprm = (aLR / pLR / gam) * pow(pstar / pLR, -0.5 * gamp1 / gam);
    return function;
}

static PressureFunction f_and_fprm_shock(PetscReal pstar, PetscReal pLR, PetscReal rhoLR, PetscReal gam, PetscReal gamm1, PetscReal gamp1){
    // compute value of pressure function for shock
    PetscReal A = 2./gamp1/rhoLR;
    PetscReal B = gamm1*pLR/gamp1;
    PetscReal sqrtterm =  PetscSqrtReal(A/(pstar+B));
    PressureFunction function;
    function.f = (pstar-pLR)*sqrtterm;
    function.fprm = sqrtterm*(1.-0.5*(pstar-pLR)/(B+pstar));
    return function;
}

#define EPS 1.e-6
#define MAXIT 100
PetscErrorCode DetermineStarState(const InitialConditions* setup, StarState* starState){
    // compute the speed of sound
    PetscReal aL = PetscSqrtReal(setup->gamma*setup->pL/setup->rhoL);
    PetscReal aR = PetscSqrtReal(setup->gamma*setup->pR/setup->rhoR);

    //first guess pstar based on two-rarefacation approximation
    starState->pstar = aL+aR - 0.5*(setup->gamma-1.)*(setup->uR-setup->uL);
    starState->pstar  = starState->pstar  / (aL/pow(setup->pL,0.5*(setup->gamma-1.)/setup->gamma) + aR/pow(setup->pR,0.5*(setup->gamma-1.)/setup->gamma) );
    starState->pstar = pow(starState->pstar,2.*setup->gamma/(setup->gamma-1.));
    starState->gamm1 = setup->gamma-1.;
    starState->gamp1 = setup->gamma+1.;

    PressureFunction fL;
    if (starState->pstar <= setup->pL){
        fL = f_and_fprm_rarefaction(starState->pstar, setup->pL,aL,setup->gamma,starState->gamm1,starState->gamp1);
    }else{
        fL = f_and_fprm_shock(starState->pstar,setup->pL,setup->rhoL,setup->gamma,starState->gamm1,starState->gamp1);
    }

    PressureFunction fR;
    if (starState->pstar <= setup->pR) {
        fR = f_and_fprm_rarefaction(starState->pstar, setup->pR, aR, setup->gamma, starState->gamm1, starState->gamp1);
    }else {
        fR = f_and_fprm_shock(starState->pstar, setup->pR, setup->rhoR, setup->gamma, starState->gamm1, starState->gamp1);
    }
    PetscReal delu = setup->uR-setup->uL;

    // iterate using Newton-Rapson
    if ((fL.f+fR.f+delu)> EPS) {
        // iterate using Newton-Rapson
        for(PetscInt it =0; it < MAXIT+4; it++){
            PetscReal pold = starState->pstar;
            starState->pstar = pold - (fL.f+fR.f+delu)/(fL.fprm+fR.fprm);

            if(starState->pstar < 0){
                starState->pstar = EPS;
            }

            if(2.0*PetscAbsReal(starState->pstar - pold)/(starState->pstar + pold) < EPS){
                break;
            }else{
                if(starState->pstar < setup->pL){
                    fL = f_and_fprm_rarefaction(starState->pstar, setup->pL,aL,setup->gamma,starState->gamm1,starState->gamp1);
                }else{
                    fL = f_and_fprm_shock(starState->pstar,setup->pL,setup->rhoL,setup->gamma,starState->gamm1,starState->gamp1);
                }
                if (starState->pstar<=setup->pR) {
                    fR = f_and_fprm_rarefaction(starState->pstar, setup->pR, aR, setup->gamma, starState->gamm1, starState->gamp1);
                }else {
                    fR = f_and_fprm_shock(starState->pstar, setup->pR, setup->rhoR, setup->gamma, starState->gamm1, starState->gamp1);
                }
            }

            if (it>MAXIT){
                SETERRQ(PETSC_COMM_WORLD,1,"error in Riemann.find_pstar - did not converage for pstar" );
            }
        }
    }

    // determine rest of star state
    starState->ustar = 0.5*(setup->uL+setup->uR+fR.f-fL.f);

    // left star state
    PetscReal pratio = starState->pstar/setup->pL;
    if (starState->pstar<=setup->pL) {  // rarefaction
        starState->rhostarL = setup->rhoL * PetscPowReal(pratio, 1. / setup->gamma);
        starState->astarL = aL * PetscPowReal(pratio, 0.5 * starState->gamm1 / setup->gamma);
        starState->SHL = setup->uL - aL;
        starState->STL = starState->ustar - starState->astarL;
    }else {  // #shock
        starState->rhostarL = setup->rhoL * (pratio + starState->gamm1 / starState->gamp1) / (starState->gamm1 * pratio / starState->gamp1 + 1.);
        starState->SL = setup->uL - aL * PetscSqrtReal(0.5 * starState->gamp1 / setup->gamma * pratio + 0.5 * starState->gamm1 / setup->gamma);
    }

    // right star state
    pratio = starState->pstar/setup->pR;
    if (starState->pstar<=setup->pR) {  // # rarefaction
        starState->rhostarR = setup->rhoR * PetscPowReal(pratio, 1. / setup->gamma);
        starState->astarR = aR * PetscPowReal(pratio, 0.5 * starState->gamm1 / setup->gamma);
        starState-> SHR = setup->uR + aR;
        starState->STR = starState->ustar + starState->astarR;
    }else {  // shock
        starState->rhostarR = setup->rhoR * (pratio + starState->gamm1 / starState->gamp1) / (starState->gamm1 * pratio / starState->gamp1 + 1.);
        starState->SR = setup->uR + aR * PetscSqrtReal(0.5 * starState->gamp1 / setup->gamma * pratio + 0.5 * starState->gamm1 / setup->gamma);
    }
    return 0;
}

void SetExactSolutionAtPoint(PetscInt dim, PetscReal xDt, const InitialConditions* setup, const StarState* starState, EulerNode* uu){
    PetscReal p;
    // compute the speed of sound
    PetscReal aL = PetscSqrtReal(setup->gamma*setup->pL/setup->rhoL);
    PetscReal aR = PetscSqrtReal(setup->gamma*setup->pR/setup->rhoR);

    for(PetscInt i =0; i < dim; i++){
        uu->rhoU[i] = 0.0;
    }

    if (xDt <= starState->ustar) {  //# left of contact surface
        if (starState->pstar <= setup->pL) {  // # rarefaction
            if (xDt <= starState->SHL) {
                uu->rho = setup->rhoL;
                p = setup->pL;
                uu->rhoU[0] = setup->uL*uu->rho;
            }else if (xDt <=starState->STL) {  //#SHL < x / t < STL
                PetscReal tmp = 2. / starState->gamp1 + (starState->gamm1 / starState->gamp1 / aL) * (setup->uL - xDt);
                uu->rho = setup->rhoL * pow(tmp, 2. / starState->gamm1);
                uu->rhoU[0] = uu->rho * (2. / starState->gamp1) * (aL + 0.5 * starState->gamm1 * setup->uL + xDt);
                p = setup->pL * pow(tmp, 2. * setup->gamma / starState->gamm1);
            }else {  //# STL < x/t < u*
                uu->rho = starState->rhostarL;
                p = starState->pstar;
                uu->rhoU[0] = uu->rho * starState->ustar;
            }
        }else{  //# shock
            if (xDt<= starState->SL) {  // # xDt < SL
                uu->rho = setup->rhoL;
                p = setup->pL;
                uu->rhoU[0] = uu->rho * setup->uL;
            }else {  //# SL < xDt < ustar
                uu->rho = starState->rhostarL;
                p = starState->pstar;
                uu->rhoU[0] = uu->rho * starState->ustar;
            }
        }
    }else{//# right of contact surface
        if (starState->pstar<=setup->pR) {  //# rarefaction
            if (xDt>= starState->SHR) {
                uu->rho = setup->rhoR;
                p = setup->pR;
                uu->rhoU[0] = uu->rho * setup->uR;
            }else if (xDt >= starState->STR) {  // # SHR < x/t < SHR
                PetscReal tmp = 2./starState->gamp1 - (starState->gamm1/starState->gamp1/aR)*(setup->uR-xDt);
                uu->rho = setup->rhoR*PetscPowReal(tmp,2./starState->gamm1);
                uu->rhoU[0] = uu->rho * (2./starState->gamp1)*(-aR + 0.5*starState->gamm1*setup->uR+xDt);
                p = setup->pR*PetscPowReal(tmp,2.*setup->gamma/starState->gamm1);
            }else{ //# u* < x/t < STR
                uu->rho = starState->rhostarR;
                p = starState->pstar;
                uu->rhoU[0] = uu->rho * starState->ustar;
            }
        }else {//# shock
            if (xDt>= starState->SR) {  // # xDt > SR
                uu->rho = setup->rhoR;
                p = setup->pR;
                uu->rhoU[0] = uu->rho * setup->uR;
            }else {//#ustar < xDt < SR
                uu->rho = starState->rhostarR;
                p = starState->pstar;
                uu->rhoU[0] = uu->rho * starState->ustar;
            }
        }
    }
    PetscReal e =  p/starState->gamm1/uu->rho;
    PetscReal E = e + 0.5*(uu->rhoU[0]/uu->rho)*(uu->rhoU[0]/uu->rho);
    uu->rhoE = uu->rho*E;
}


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
    ierr = MPI_Comm_rank(PetscObjectComm((PetscObject)dm), &rank);CHKERRMPI(ierr);
    ierr = MPI_Comm_size(PetscObjectComm((PetscObject)dm), &size);CHKERRMPI(ierr);

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
                }else{
                    printf("ghostNode %d\n", c);
                }
            }

            fclose(fptr);
        }
        MPI_Barrier(PetscObjectComm((PetscObject)dm));
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
    PetscErrorCode     (*func[3]) (PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar *u, void *ctx) = {SetExactSolutionRho, SetExactSolutionRhoU, SetExactSolutionRhoE};
    void* ctxs[3] ={ctx, ctx, ctx};
    ierr    = DMProjectFunction(dm,time,func,ctxs,INSERT_ALL_VALUES,e);CHKERRQ(ierr);

    // just print to a file for now
    ierr = PrintVector(dm, e, step, "exact");CHKERRQ(ierr);
    ierr = PrintVector(dm, u, step, "solution");CHKERRQ(ierr);

    PetscPrintf(PETSC_COMM_WORLD, "TS %d: %f\n", step, time);

    Vec g;
    ierr =  VecGhostGetLocalForm(u,&g);CHKERRQ(ierr);

    VecView(g, PETSC_VIEWER_STDOUT_SELF);
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

static bcFunc(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar bcval[]){
    puts("test");
}

void bcFunc2(PetscInt dim, PetscInt Nf, PetscInt NfAux,
const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
PetscReal t, const PetscReal x[], const PetscReal n[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[]){
    puts("here2");
}

/* PetscReal* => EulerNode* conversion */
static PetscErrorCode PhysicsBoundary_Euler_Left(PetscReal time, const PetscReal *c, const PetscReal *n, const PetscScalar *a_xI, PetscScalar *a_xG, void *ctx)
{
    const EulerNode *xI = (const EulerNode*)a_xI;
    EulerNode       *xG = (EulerNode*)a_xG;
    ProblemSetup* prob = (ProblemSetup*)ctx;
    PetscFunctionBeginUser;
    xG->rho = 14.2;//prob->initialConditions.rhoL;
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
    PetscInt nx[] = {10, 1};
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
    problem.initialConditions.maxTime = 0.25;
    problem.initialConditions.length = 1;
    problem.initialConditions.gamma = 1.4;

    // case 2
//    problem.initialConditions.rhoL=1.0;
//    problem.initialConditions.uL=-2.0;
//    problem.initialConditions.pL=0.4;
//    problem.initialConditions.rhoR=1.0;
//    problem.initialConditions.uR=2.0;
//    problem.initialConditions.pR=0.4;
//    problem.initialConditions.maxTime = 0.15;
//    problem.initialConditions.length = 1;
//    problem.initialConditions.gamma = 1.4;

    // Compute the star state
    ierr = DetermineStarState(&problem.initialConditions, &problem.starState);CHKERRQ(ierr);

    // Setup the flow data
    FlowData flowData;     /* store some of the flow data*/
    ierr = FlowCreate(&flowData);CHKERRQ(ierr);

    //Setup
    CompressibleFlow_SetupDiscretization(flowData, dm);

    // Add in the flow parameters
    PetscScalar params[TOTAL_COMPRESSIBLE_FLOW_PARAMETERS];
    params[CFL] = 1.0;
    params[GAMMA] = problem.initialConditions.gamma;

    // set up the finite volume fluxes
    CompressibleFlow_StartProblemSetup(flowData, TOTAL_COMPRESSIBLE_FLOW_PARAMETERS, params);
    DMView(flowData->dm, PETSC_VIEWER_STDERR_SELF);
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

//    const PetscInt idsTest[] = {4};
//    ierr = PetscDSAddBoundary(prob, DM_BC_NATURAL , "test", "Face Sets", 0, 0, NULL, (void (*)(void)) PhysicsBoundary_Euler_Left, NULL, 1, idsTest, &problem);CHKERRQ(ierr);
//    ierr = PetscDSSetBdResidual(prob, 0, bcFunc2, NULL);CHKERRQ(ierr);
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