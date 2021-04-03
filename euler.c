#include <petsc.h>
#include "flow.h"
#include "mesh.h"

#define DIM 2                   /* Geometric dimension */
#define NFIELDS 3
static const FlowFieldDescriptor PhysicsFields[] = {{"Density", "Den",1},{"Momentum", "Momentum",DIM},{"Energy", "Energy",1},{NULL,0}};
typedef enum {EULER_PAR_GAMMA,EULER_PAR_RHOR,EULER_PAR_AMACH,EULER_PAR_ITANA,EULER_PAR_SIZE} EulerParamIdx;

typedef struct {
    PetscReal rho;
    PetscReal rhoU[DIM];
    PetscReal rhoE;
} EulerNode;
typedef union {
    EulerNode eulernode;
    PetscReal vals[DIM+2];
} EulerNodeUnion;

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
} Setup;

typedef struct {
    PetscReal pstar,ustar,rhostarL,astarL,SL,SHL,STL,rhostarR,astarR,SR,SHR,STR,gamm1,gamp1;
} StarState;

typedef struct {
    StarState starState;
    Setup setup;
    PetscReal cfl;
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
static PetscErrorCode DetermineStarState(const Setup* setup, StarState* starState){
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

static void SetExactSolutionAtPoint(PetscInt dim, PetscReal xDt, const Setup* setup, const StarState* starState, EulerNode* uu){
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

    PetscReal xDt = (x[0]-prob->setup.length/2)/time;
    EulerNode uu;
    SetExactSolutionAtPoint(dim, xDt, &prob->setup, &prob->starState, &uu);

    u[0] = uu.rho;
    return 0;
}

static PetscErrorCode SetExactSolutionRhoU(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar *u, void *ctx){
    ProblemSetup* prob = (ProblemSetup*)ctx;

    PetscReal xDt = (x[0]-prob->setup.length/2)/time;
    EulerNode uu;
    SetExactSolutionAtPoint(dim, xDt, &prob->setup, &prob->starState, &uu);
    u[0] = uu.rhoU[0];
    u[1] = uu.rhoU[1];
    return 0;
}

static PetscErrorCode SetExactSolutionRhoE(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar *u, void *ctx){
    ProblemSetup* prob = (ProblemSetup*)ctx;

    PetscReal xDt = (x[0]-prob->setup.length/2)/time;
    EulerNode uu;
    SetExactSolutionAtPoint(dim, xDt, &prob->setup, &prob->starState, &uu);
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

    FILE *fptr = fopen(filename, "w");
    fprintf(fptr, "x rho u e\n");
    for (PetscInt c = cStart; c < cEnd; ++c) {
        PetscFVCellGeom       *cg;
        const EulerNode           *xc;

        ierr = DMPlexPointLocalRead(dmCell, c, cgeom, &cg);CHKERRQ(ierr);
        ierr = DMPlexPointGlobalFieldRead(dm, c, 0, x, &xc);CHKERRQ(ierr);
        PetscReal u0 = xc->rhoU[0]/xc->rho;
        fprintf(fptr, "%f %f %f %f\n", cg->centroid[0], xc->rho, u0, (xc->rhoE/xc->rho)-0.5*u0*u0);
    }

    fclose(fptr);
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
    xG->rho = prob->setup.rhoL;
    PetscReal eT = prob->setup.rhoL*((prob->setup.pL /(prob->setup.gamma -1) / prob->setup.rhoL) + 0.5 * prob->setup.uL * prob->setup.uL);
    xG->rhoE = eT;
    xG->rhoU[0] = prob->setup.rhoL * prob->setup.uL;
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
    xG->rho = prob->setup.rhoR;
    PetscReal eT = prob->setup.rhoR*((prob->setup.pR /(prob->setup.gamma -1)/ prob->setup.rhoR) + 0.5 * prob->setup.uR * prob->setup.uR);
    xG->rhoE = eT;
    xG->rhoU[0] = prob->setup.rhoR * prob->setup.uR;
    xG->rhoU[1] = 0.0;

    PetscFunctionReturn(0);
}

static PetscErrorCode  ComputeTimeStep(TS ts){
    DM dm;
    PetscErrorCode ierr = TSGetDM(ts, &dm);CHKERRQ(ierr);
    Vec v;
    TSGetSolution(ts, &v);
    ProblemSetup *problem;
    ierr = DMGetApplicationContext(dm, &problem);CHKERRQ(ierr);


    Vec                cellgeom;
    ierr = DMPlexGetGeometryFVM(dm, NULL, &cellgeom, NULL);CHKERRQ(ierr);
    PetscInt cStart, cEnd;
    ierr = DMPlexGetSimplexOrBoxCells(dm, 0, &cStart, &cEnd);CHKERRQ(ierr);
    DM dmCell;
    ierr = VecGetDM(cellgeom, &dmCell);CHKERRQ(ierr);
    const PetscScalar *cgeom;
    ierr = VecGetArrayRead(cellgeom, &cgeom);CHKERRQ(ierr);
    const PetscScalar      *x;
    ierr = VecGetArrayRead(v, &x);CHKERRQ(ierr);

    PetscReal dtMin = 1.0;

    for (PetscInt c = cStart; c < cEnd; ++c) {
        PetscFVCellGeom       *cg;
        const EulerNode           *xc;

        ierr = DMPlexPointLocalRead(dmCell, c, cgeom, &cg);CHKERRQ(ierr);
        ierr = DMPlexPointGlobalFieldRead(dm, c, 0, x, &xc);CHKERRQ(ierr);

        PetscReal rho = xc->rho;
        PetscReal u = xc->rhoU[0]/rho;
        PetscReal e = (xc->rhoE / rho) - 0.5 * u * u;
        PetscReal p = (problem->setup.gamma-1) * rho * e;

        PetscReal a = PetscSqrtReal(problem->setup.gamma*p/rho);
        PetscReal dt = problem->cfl*cg->volume/(a + PetscAbsReal(u));
        dtMin = PetscMin(dtMin, dt);
    }

    printf("TimeStep: %f\n", dtMin);
    ierr = TSSetTimeStep(ts, dtMin);CHKERRQ(ierr);

    ierr = VecRestoreArrayRead(cellgeom, &cgeom);CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(v, &x);CHKERRQ(ierr);
    return 0;
}

static void ComputeFluxRho(PetscInt dim, PetscInt Nf, const PetscReal *qp, const PetscReal *n, const EulerNode *xL, const EulerNode *xR, PetscInt numConstants, const PetscScalar constants[], PetscReal *flux, void* ctx) {
//    dim	- The spatial dimension
//    Nf	- The number of fields
//    x	- The coordinates at a point on the interface
//    n	- The normal vector to the interface
//    uL	- The state vector to the left of the interface
//    uR	- The state vector to the right of the interface
//    flux	- output array of flux through the interface
//    numConstants	- number of constant parameters
//    constants	- constant parameters
//    ctx	- optional user context
    ProblemSetup* prob = (ProblemSetup*)ctx;

    // this is a hack, only add in flux from left/right
//    if(PetscAbs(n[0]) > 1E-5) {
        // Setup Godunov
        Setup currentValues;

        currentValues.gamma = prob->setup.gamma;
        currentValues.length = prob->setup.length;

        currentValues.rhoL = xL->rho;
        currentValues.uL = xL->rhoU[0] / currentValues.rhoL;
        PetscReal eL = (xL->rhoE / currentValues.rhoL) - 0.5 * currentValues.uL * currentValues.uL;
        currentValues.pL = (prob->setup.gamma-1) * currentValues.rhoL * eL;

        currentValues.rhoR = xR->rho;
        currentValues.uR = xR->rhoU[0] / currentValues.rhoR;
        PetscReal eR = (xR->rhoE / currentValues.rhoR) - 0.5 * currentValues.uR * currentValues.uR;
        currentValues.pR = (prob->setup.gamma-1) * currentValues.rhoR * eR;

        StarState result;
        DetermineStarState(&currentValues, &result);
        EulerNode exact;
        SetExactSolutionAtPoint(dim, 0.0, &currentValues, &result, &exact);


        PetscReal rho = exact.rho;
        PetscReal u = exact.rhoU[0]/rho;
        PetscReal e = (exact.rhoE / rho) - 0.5 * u * u;
        PetscReal p = (prob->setup.gamma-1) * rho * e;

        flux[0] = (rho * u) * PetscSignReal(n[0]);
//        flux->rhoU[0] = (rho * u * u + p)* PetscSignReal(n[0]);
//        flux->rhoU[1] = 0.0;
//        PetscReal et = e + 0.5 * u * u;
//        flux->rhoE = (rho * u * (et + p / rho))* PetscSignReal(n[0]);

//        printf("%f,%f %f %f,%f\n", qp[0], qp[1], flux[0], n[0], n[1]);mm

//    }else{
//        flux[0] = 0.0;
////        flux->rhoU[0] =0.0;
////        flux->rhoU[1] = 0.0;
////        flux->rhoE = 0.0;
//
//    }

//    F[0][i]=rho[i]*u[i]
//    F[1][i]=rho[i]*u[i]*u[i]+p[i]
//    et = e[i]+0.5*u[i]*u[i]
//    F[2][i]=rho[i]*u[i]*(et+p[i]/rho[i])

}


static void ComputeFluxU(PetscInt dim, PetscInt Nf, const PetscReal *qp, const PetscReal *n, const EulerNode *xL, const EulerNode *xR, PetscInt numConstants, const PetscScalar constants[], PetscReal *flux, void* ctx) {
//    dim	- The spatial dimension
//    Nf	- The number of fields
//    x	- The coordinates at a point on the interface
//    n	- The normal vector to the interface
//    uL	- The state vector to the left of the interface
//    uR	- The state vector to the right of the interface
//    flux	- output array of flux through the interface
//    numConstants	- number of constant parameters
//    constants	- constant parameters
//    ctx	- optional user context
    ProblemSetup* prob = (ProblemSetup*)ctx;

    // this is a hack, only add in flux from left/right
    if(PetscAbs(n[0]) > 1E-5) {
        // Setup Godunov
        Setup currentValues;

        currentValues.gamma = prob->setup.gamma;
        currentValues.length = prob->setup.length;

        currentValues.rhoL = xL->rho;
        currentValues.uL = xL->rhoU[0] / currentValues.rhoL;
        PetscReal eL = (xL->rhoE / currentValues.rhoL) - 0.5 * currentValues.uL * currentValues.uL;
        currentValues.pL = (prob->setup.gamma-1) * currentValues.rhoL * eL;

        currentValues.rhoR = xR->rho;
        currentValues.uR = xR->rhoU[0] / currentValues.rhoR;
        PetscReal eR = (xR->rhoE / currentValues.rhoR) - 0.5 * currentValues.uR * currentValues.uR;
        currentValues.pR = (prob->setup.gamma-1) * currentValues.rhoR * eR;

        StarState result;
        DetermineStarState(&currentValues, &result);
        EulerNode exact;
        SetExactSolutionAtPoint(dim, 0.0, &currentValues, &result, &exact);


        PetscReal rho = exact.rho;
        PetscReal u = exact.rhoU[0]/rho;
        PetscReal e = (exact.rhoE / rho) - 0.5 * u * u;
        PetscReal p = (prob->setup.gamma-1) * rho * e;

//        flux[0] = (rho * u) * PetscSignReal(n[0]);
        flux[0] = (rho * u * u + p)* PetscSignReal(n[0]);
        flux[1] = 0.0;
//        PetscReal et = e + 0.5 * u * u;
//        flux->rhoE = (rho * u * (et + p / rho))* PetscSignReal(n[0]);

//        printf("%f,%f %f %f %f %f\n", qp[0], qp[1], flux->rho, flux->rhoU[0], flux->rhoE, n[0]);

    }else{
        flux[0] = 0.0;
        flux[1] = 0.0;
//        flux->rhoU[0] =0.0;
//        flux->rhoU[1] = 0.0;
//        flux->rhoE = 0.0;

    }

//    F[0][i]=rho[i]*u[i]
//    F[1][i]=rho[i]*u[i]*u[i]+p[i]
//    et = e[i]+0.5*u[i]*u[i]
//    F[2][i]=rho[i]*u[i]*(et+p[i]/rho[i])

}


static void ComputeFluxE(PetscInt dim, PetscInt Nf, const PetscReal *qp, const PetscReal *n, const EulerNode *xL, const EulerNode *xR, PetscInt numConstants, const PetscScalar constants[], PetscReal *flux, void* ctx) {
//    dim	- The spatial dimension
//    Nf	- The number of fields
//    x	- The coordinates at a point on the interface
//    n	- The normal vector to the interface
//    uL	- The state vector to the left of the interface
//    uR	- The state vector to the right of the interface
//    flux	- output array of flux through the interface
//    numConstants	- number of constant parameters
//    constants	- constant parameters
//    ctx	- optional user context
    ProblemSetup* prob = (ProblemSetup*)ctx;

    // this is a hack, only add in flux from left/right
    if(PetscAbs(n[0]) > 1E-5) {
        // Setup Godunov
        Setup currentValues;

        currentValues.gamma = prob->setup.gamma;
        currentValues.length = prob->setup.length;

        currentValues.rhoL = xL->rho;
        currentValues.uL = xL->rhoU[0] / currentValues.rhoL;
        PetscReal eL = (xL->rhoE / currentValues.rhoL) - 0.5 * currentValues.uL * currentValues.uL;
        currentValues.pL = (prob->setup.gamma-1) * currentValues.rhoL * eL;

        currentValues.rhoR = xR->rho;
        currentValues.uR = xR->rhoU[0] / currentValues.rhoR;
        PetscReal eR = (xR->rhoE / currentValues.rhoR) - 0.5 * currentValues.uR * currentValues.uR;
        currentValues.pR = (prob->setup.gamma-1) * currentValues.rhoR * eR;

        StarState result;
        DetermineStarState(&currentValues, &result);
        EulerNode exact;
        SetExactSolutionAtPoint(dim, 0.0, &currentValues, &result, &exact);


        PetscReal rho = exact.rho;
        PetscReal u = exact.rhoU[0]/rho;
        PetscReal e = (exact.rhoE / rho) - 0.5 * u * u;
        PetscReal p = (prob->setup.gamma-1) * rho * e;

//        flux[0] = (rho * u) * PetscSignReal(n[0]);
//        flux[0] = (rho * u * u + p)* PetscSignReal(n[0]);
//        flux[1] = 0.0;
        PetscReal et = e + 0.5 * u * u;
        flux[0] = (rho * u * (et + p / rho))* PetscSignReal(n[0]);

//        printf("%f,%f %f %f %f %f\n", qp[0], qp[1], flux->rho, flux->rhoU[0], flux->rhoE, n[0]);

    }else{
        flux[0] = 0.0;
//        flux[1] = 0.0;
//        flux->rhoU[0] =0.0;
//        flux->rhoU[1] = 0.0;
//        flux->rhoE = 0.0;

    }

//    F[0][i]=rho[i]*u[i]
//    F[1][i]=rho[i]*u[i]*u[i]+p[i]
//    et = e[i]+0.5*u[i]*u[i]
//    F[2][i]=rho[i]*u[i]*(et+p[i]/rho[i])

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

    // Setup the problem
    ProblemSetup problem;

    // case 1 - Sod problem
//    problem.setup.rhoL=1.0;
//    problem.setup.uL=0.0;
//    problem.setup.pL=1.0;
//    problem.setup.rhoR=0.125;
//    problem.setup.uR=0.0;
//    problem.setup.pR=0.1;
//    problem.setup.maxTime = 0.25;
//    problem.setup.length = 1;
//    problem.setup.gamma = 1.4;
//    problem.cfl = .5;

    // case 2 - 123 problem - expansion left and expansion right
    problem.setup.rhoL=1.0;
    problem.setup.uL=-2.0;
    problem.setup.pL=0.4;
    problem.setup.rhoR=1.0;
    problem.setup.uR=2.0;
    problem.setup.pR=0.4;
    problem.setup.maxTime = 0.15;
    problem.setup.length = 1;
    problem.setup.gamma = 1.4;
    problem.cfl = 0.5;

    ierr = TSCreate(PETSC_COMM_WORLD, &ts);
    CHKERRABORT(PETSC_COMM_WORLD, ierr);
    ierr = TSSetType(ts, TSCN);CHKERRQ(ierr);

    //PetscErrorCode DMPlexCreateBoxMesh(MPI_Comm comm, PetscInt dim, PetscBool simplex, const PetscInt faces[], const PetscReal lower[], const PetscReal upper[], const DMBoundaryType periodicity[], PetscBool interpolate, DM *dm)
    PetscReal start[] = {0.0, 0.0};
    PetscReal end[] = {problem.setup.length, 1};
    PetscInt nx[] = {100, 1};
    DMBoundaryType bcType[] = {DM_BOUNDARY_NONE, DM_BOUNDARY_NONE};
    ierr = DMPlexCreateBoxMesh(PETSC_COMM_WORLD, DIM, PETSC_FALSE, nx, start, end, bcType, PETSC_TRUE, &dm);CHKERRQ(ierr);
    puts("after DMPlexCreateBoxMesh Call");
    DMView(dm, PETSC_VIEWER_STDOUT_WORLD);

    ierr = DMSetFromOptions(dm);CHKERRQ(ierr);

    {
        DM gdm;
        ierr = DMPlexConstructGhostCells(dm, NULL, NULL, &gdm);CHKERRQ(ierr);
        ierr = DMDestroy(&dm);CHKERRQ(ierr);
        dm   = gdm;
    }
    ierr = DMView(dm, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    ierr = DMViewFromOptions(dm, NULL, "-dm_view");CHKERRQ(ierr);

    DMLabel label;
    ierr = DMGetLabel(dm, "marker", &label );CHKERRQ(ierr);
    ierr = DMLabelView(label, PETSC_VIEWER_STDOUT_WORLD);;CHKERRQ(ierr);

    CHKERRABORT(PETSC_COMM_WORLD, ierr);
    ierr = TSSetDM(ts, dm);
    CHKERRABORT(PETSC_COMM_WORLD, ierr);
    ierr = TSSetExactFinalTime(ts, TS_EXACTFINALTIME_MATCHSTEP);
    CHKERRABORT(PETSC_COMM_WORLD, ierr);

    // setup the FV

    {// Setup the fields
        PetscInt f, dof;
        for (f=0,dof=0; f < 3; f++) {
            PetscInt newDof = PhysicsFields[f].components;

//            if (newDof == 1) {
                PetscFV           fvm;
                ierr = PetscFVCreate(PETSC_COMM_WORLD, &fvm);CHKERRQ(ierr);

                ierr = PetscFVSetFromOptions(fvm);CHKERRQ(ierr);
                ierr = PetscFVSetNumComponents(fvm, newDof);CHKERRQ(ierr);
                ierr = PetscFVSetSpatialDimension(fvm, DIM);CHKERRQ(ierr);
                ierr = PetscObjectSetName((PetscObject) fvm,PhysicsFields[f].fieldName);CHKERRQ(ierr);

                /* FV is now structured with one field having all physics as components */
                ierr = DMAddField(dm, NULL, (PetscObject) fvm);CHKERRQ(ierr);
//            }
//            else {
//                PetscInt j;
//
//                for (j = 0; j < newDof; j++) {
//                    char     compName[256]  = "Unknown";
//
//                    ierr = PetscSNPrintf(compName,sizeof(compName),"%s_%d",PhysicsFields[f].fieldName,j);CHKERRQ(ierr);
//                    ierr = PetscFVSetComponentName(fvm,dof+j,compName);CHKERRQ(ierr);
//                }
//            }
//            dof += newDof;
        }
    }

    // Compute the star state
    ierr = DetermineStarState(&problem.setup, &problem.starState);CHKERRQ(ierr);



    PetscDS           prob;
    ierr = DMCreateDS(dm);CHKERRQ(ierr);
    ierr = DMGetDS(dm, &prob);CHKERRQ(ierr);
    ierr = DMSetApplicationContext(dm, &problem);CHKERRQ(ierr);

    //TODO: Add flux
    ierr = PetscDSSetRiemannSolver(prob, 0,ComputeFluxRho);CHKERRQ(ierr);
    ierr = PetscDSSetContext(prob, 0, &problem);CHKERRQ(ierr);
    ierr = PetscDSSetRiemannSolver(prob, 1,ComputeFluxU);CHKERRQ(ierr);
    ierr = PetscDSSetContext(prob, 1, &problem);CHKERRQ(ierr);
    ierr = PetscDSSetRiemannSolver(prob, 2,ComputeFluxE);CHKERRQ(ierr);
    ierr = PetscDSSetContext(prob, 2, &problem);CHKERRQ(ierr);
    ierr = PetscDSSetFromOptions(prob);CHKERRQ(ierr);
    //TODO: Apply boundary

    // setup the solution vector, this olds everything
    Vec X;
    ierr = DMCreateGlobalVector(dm, &X);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject) X, "solution");CHKERRQ(ierr);


//    ierr = DMTSSetBoundaryLocal(dm, DMPlexTSComputeBoundary, NULL);CHKERRQ(ierr);
//    ierr = DMTSSetIFunctionLocal(dm, DMPlexTSComputeIFunctionFEM, NULL);CHKERRQ(ierr);
//    ierr = DMTSSetIJacobianLocal(dm, DMPlexTSComputeIJacobianFEM, NULL);CHKERRQ(ierr);
//    ierr = DMTSSetRHSFunctionLocal(dm, DMPlexTSComputeRHSFunctionFVM, NULL);CHKERRQ(ierr);//TODO: This is were we set the RHS function

    const PetscInt idsLeft[]= {4};
    ierr = PetscDSAddBoundary(prob, DM_BC_NATURAL_RIEMANN, "wall left", "Face Sets", 0, 0, NULL, (void (*)(void)) PhysicsBoundary_Euler_Left, NULL, 1, idsLeft, &problem);CHKERRQ(ierr);
    const PetscInt idsRight[]= {2};
    ierr = PetscDSAddBoundary(prob, DM_BC_NATURAL_RIEMANN, "wall right", "Face Sets", 0, 0, NULL, (void (*)(void)) PhysicsBoundary_Euler_Right, NULL, 1, idsRight, &problem);CHKERRQ(ierr);

    const PetscInt mirror[]= {1, 3};
    ierr = PetscDSAddBoundary(prob, DM_BC_NATURAL_RIEMANN, "mirrorWall", "Face Sets", 0, 0, NULL, (void (*)(void)) PhysicsBoundary_Euler_Mirror, NULL, 2, mirror, &problem);CHKERRQ(ierr);


//    ierr = DMTSSetBoundaryLocal(dm, DMPlexTSComputeBoundary, NULL);CHKERRQ(ierr);
    ierr = DMTSSetRHSFunctionLocal(dm, DMPlexTSComputeRHSFunctionFVM, NULL);CHKERRQ(ierr);//TODO: This is were we set the RHS function
    ierr = TSSetMaxTime(ts,problem.setup.maxTime);CHKERRQ(ierr);
    ierr = TSMonitorSet(ts, MonitorError, &problem, NULL);CHKERRQ(ierr);
    ierr = TSSetTimeStep(ts, 0.0008);CHKERRQ(ierr);
    ierr = TSSetFromOptions(ts);CHKERRQ(ierr);
    ierr = TSSetPostStep(ts, ComputeTimeStep);CHKERRQ(ierr);

    // set the initial conditions
    PetscErrorCode     (*func[3]) (PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar *u, void *ctx) = {SetExactSolutionRho, SetExactSolutionRhoU, SetExactSolutionRhoE};
    void* ctxs[3] ={&problem, &problem, &problem};
    ierr    = DMProjectFunction(dm,0.0,func,ctxs,INSERT_ALL_VALUES,X);CHKERRQ(ierr);



    ierr = TSSolve(ts,X);CHKERRQ(ierr);
//    ierr = TSGetSolveTime(ts,&ftime);CHKERRQ(ierr);
//    ierr = TSGetStepNumber(ts,&nsteps);
    //    CHKERRQ(ierr);


    return PetscFinalize();

}