#include <petsc.h>
#include <petscmath.h>
#include "compressibleFlow.h"
#include "mesh.h"
#include "petscdmplex.h"
#include "petscts.h"

//MMS from Verification of a Compressible CFD Code using the Method of Manufactured Solutions, Christopher J. Roy,† Thomas M. Smith,‡ and Curtis C. Ober§

// Define
#define Pi PETSC_PI
#define Sin PetscSinReal
#define Cos PetscCosReal
#define Power PetscPowReal

typedef struct {
    PetscReal phiO;
    PetscReal phiX;
    PetscReal phiY;
    PetscReal phiZ;
    PetscReal aPhiX;
    PetscReal aPhiY;
    PetscReal aPhiZ;
} PhiConstants;

typedef struct {
    PetscInt dim;
    PhiConstants rho;
    PhiConstants u;
    PhiConstants v;
    PhiConstants w;
    PhiConstants p;
    PetscReal L;
    PetscReal gamma;
    PetscReal R;
    PetscReal mu;
} Constants;

typedef struct {
    Constants constants;
    FlowData flowData;
} ProblemSetup;

static PetscErrorCode RhoExact(PetscInt dim, PetscReal time, const PetscReal xyz[], PetscInt Nf, PetscScalar *u, void *ctx) {
    Constants *constants = (Constants *)ctx;
    PetscReal rhoO = constants->rho.phiO;
    PetscReal rhoX = constants->rho.phiX;
    PetscReal rhoY = constants->rho.phiY;
    PetscReal rhoZ = constants->rho.phiZ;
    PetscReal aRhoX = constants->rho.aPhiX;
    PetscReal aRhoY = constants->rho.aPhiY;
    PetscReal aRhoZ = constants->rho.aPhiZ;
    PetscReal L = constants->L;

    PetscReal x = xyz[0];
    PetscReal y = xyz[1];
    PetscReal z = dim > 2? xyz[2] : 0.0;

    u[0] = rhoO + rhoY*Cos((aRhoY*Pi*y)/L) + rhoX*Sin((aRhoX*Pi*x)/L) + rhoZ*Sin((aRhoZ*Pi*z)/L);
    PetscFunctionReturn(0);
}

static PetscErrorCode RhoExactTimeDerivative(PetscInt dim, PetscReal time, const PetscReal xyz[], PetscInt Nf, PetscScalar *u, void *ctx) {
    PetscFunctionBeginUser;
    u[0] = 0.0;
    PetscFunctionReturn(0);
}
static PetscErrorCode RhoUExactTimeDerivative(PetscInt dim, PetscReal time, const PetscReal xyz[], PetscInt Nf, PetscScalar *u, void *ctx) {
    PetscFunctionBeginUser;
    for(PetscInt d =0; d < dim; d++){
        u[d] = 0.0;
    }
    PetscFunctionReturn(0);
}
static PetscErrorCode RhoEExactTimeDerivative(PetscInt dim, PetscReal time, const PetscReal xyz[], PetscInt Nf, PetscScalar *u, void *ctx) {
    PetscFunctionBeginUser;
    u[0] = 0.0;
    PetscFunctionReturn(0);
}

static PetscErrorCode RhoUExact(PetscInt dim, PetscReal time, const PetscReal xyz[], PetscInt Nf, PetscScalar *u, void *ctx) {
    PetscFunctionBeginUser;

    Constants *constants = (Constants *)ctx;
    PetscReal L = constants->L;

    PetscReal rhoO = constants->rho.phiO;
    PetscReal rhoX = constants->rho.phiX;
    PetscReal rhoY = constants->rho.phiY;
    PetscReal rhoZ = constants->rho.phiZ;
    PetscReal aRhoX = constants->rho.aPhiX;
    PetscReal aRhoY = constants->rho.aPhiY;
    PetscReal aRhoZ = constants->rho.aPhiZ;

    PetscReal uO = constants->u.phiO;
    PetscReal uX = constants->u.phiX;
    PetscReal uY = constants->u.phiY;
    PetscReal uZ = constants->u.phiZ;
    PetscReal aUX = constants->u.aPhiX;
    PetscReal aUY = constants->u.aPhiY;
    PetscReal aUZ = constants->u.aPhiZ;

    PetscReal vO = constants->v.phiO;
    PetscReal vX = constants->v.phiX;
    PetscReal vY = constants->v.phiY;
    PetscReal vZ = constants->v.phiZ;
    PetscReal aVX = constants->v.aPhiX;
    PetscReal aVY = constants->v.aPhiY;
    PetscReal aVZ = constants->v.aPhiZ;

    PetscReal wO = constants->w.phiO;
    PetscReal wX = constants->w.phiX;
    PetscReal wY = constants->w.phiY;
    PetscReal wZ = constants->w.phiZ;
    PetscReal aWX = constants->w.aPhiX;
    PetscReal aWY = constants->w.aPhiY;
    PetscReal aWZ = constants->w.aPhiZ;

    PetscReal x = xyz[0];
    PetscReal y = xyz[1];
    PetscReal z = dim > 2? xyz[2] : 0.0;

    u[0] = (uO + uY*Cos((aUY*Pi*y)/L) + uZ*Cos((aUZ*Pi*z)/L) + uX*Sin((aUX*Pi*x)/L))*
           (rhoO + rhoY*Cos((aRhoY*Pi*y)/L) + rhoX*Sin((aRhoX*Pi*x)/L) + rhoZ*Sin((aRhoZ*Pi*z)/L));
    u[1] = (rhoO + rhoY*Cos((aRhoY*Pi*y)/L) + rhoX*Sin((aRhoX*Pi*x)/L) + rhoZ*Sin((aRhoZ*Pi*z)/L))*
           (vO + vX*Cos((aVX*Pi*x)/L) + vY*Sin((aVY*Pi*y)/L) + vZ*Sin((aVZ*Pi*z)/L));

    if(dim > 2){
        u[2] = (wO + wZ*Cos((aWZ*Pi*z)/L) + wX*Sin((aWX*Pi*x)/L) + wY*Sin((aWY*Pi*y)/L))*
               (rhoO + rhoY*Cos((aRhoY*Pi*y)/L) + rhoX*Sin((aRhoX*Pi*x)/L) + rhoZ*Sin((aRhoZ*Pi*z)/L));
    }
    PetscFunctionReturn(0);
}

static PetscErrorCode RhoEExact(PetscInt dim, PetscReal time, const PetscReal xyz[], PetscInt Nf, PetscScalar *u, void *ctx) {
    PetscFunctionBeginUser;

    Constants *constants = (Constants *)ctx;
    PetscReal L = constants->L;
    PetscReal gamma = constants->gamma;

    PetscReal rhoO = constants->rho.phiO;
    PetscReal rhoX = constants->rho.phiX;
    PetscReal rhoY = constants->rho.phiY;
    PetscReal rhoZ = constants->rho.phiZ;
    PetscReal aRhoX = constants->rho.aPhiX;
    PetscReal aRhoY = constants->rho.aPhiY;
    PetscReal aRhoZ = constants->rho.aPhiZ;

    PetscReal uO = constants->u.phiO;
    PetscReal uX = constants->u.phiX;
    PetscReal uY = constants->u.phiY;
    PetscReal uZ = constants->u.phiZ;
    PetscReal aUX = constants->u.aPhiX;
    PetscReal aUY = constants->u.aPhiY;
    PetscReal aUZ = constants->u.aPhiZ;

    PetscReal vO = constants->v.phiO;
    PetscReal vX = constants->v.phiX;
    PetscReal vY = constants->v.phiY;
    PetscReal vZ = constants->v.phiZ;
    PetscReal aVX = constants->v.aPhiX;
    PetscReal aVY = constants->v.aPhiY;
    PetscReal aVZ = constants->v.aPhiZ;

    PetscReal wO = constants->w.phiO;
    PetscReal wX = constants->w.phiX;
    PetscReal wY = constants->w.phiY;
    PetscReal wZ = constants->w.phiZ;
    PetscReal aWX = constants->w.aPhiX;
    PetscReal aWY = constants->w.aPhiY;
    PetscReal aWZ = constants->w.aPhiZ;

    PetscReal pO = constants->p.phiO;
    PetscReal pX = constants->p.phiX;
    PetscReal pY = constants->p.phiY;
    PetscReal pZ = constants->p.phiZ;
    PetscReal aPX = constants->p.aPhiX;
    PetscReal aPY = constants->p.aPhiY;
    PetscReal aPZ = constants->p.aPhiZ;

    PetscReal x = xyz[0];
    PetscReal y = xyz[1];
    PetscReal z = dim > 2? xyz[2] : 0.0;

    u[0] = (rhoO + rhoY*Cos((aRhoY*Pi*y)/L) + rhoX*Sin((aRhoX*Pi*x)/L) + rhoZ*Sin((aRhoZ*Pi*z)/L))*((pO + pX*Cos((aPX*Pi*x)/L) + pZ*Cos((aPZ*Pi*z)/L) + pY*Sin((aPY*Pi*y)/L))/((-1. + gamma)*(rhoO + rhoY*Cos((aRhoY*Pi*y)/L) + rhoX*Sin((aRhoX*Pi*x)/L) + rhoZ*Sin((aRhoZ*Pi*z)/L))) +
                                                                                                    (Power(uO + uY*Cos((aUY*Pi*y)/L) + uZ*Cos((aUZ*Pi*z)/L) + uX*Sin((aUX*Pi*x)/L),2) + Power(wO + wZ*Cos((aWZ*Pi*z)/L) + wX*Sin((aWX*Pi*x)/L) + wY*Sin((aWY*Pi*y)/L),2) + Power(vO + vX*Cos((aVX*Pi*x)/L) + vY*Sin((aVY*Pi*y)/L) + vZ*Sin((aVZ*Pi*z)/L),2))/2.);
    PetscFunctionReturn(0);
}

static PetscErrorCode MonitorError(TS ts, PetscInt step, PetscReal time, Vec u, void *ctx) {
    PetscFunctionBeginUser;
    PetscErrorCode     ierr;

    // Get the DM
    DM dm;
    ierr = TSGetDM(ts, &dm);CHKERRQ(ierr);

    ierr = VecViewFromOptions(u, NULL, "-sol_view");CHKERRQ(ierr);
    ierr = PetscPrintf(PetscObjectComm((PetscObject)dm), "TS at %f\n", time);CHKERRQ(ierr);

    // Compute the error
    void            *exactCtxs[3];
    PetscErrorCode (*exactFuncs[3])(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar *u, void *ctx);
    PetscDS          ds;
    ierr = DMGetDS(dm, &ds);CHKERRQ(ierr);
    for (PetscInt f = 0; f < 3; ++f) {ierr = PetscDSGetExactSolution(ds, f, &exactFuncs[f], &exactCtxs[f]);CHKERRQ(ierr);}

    // Compute the L2 Difference
    PetscReal        ferrors[3];
    ierr = DMComputeL2FieldDiff(dm, time, exactFuncs, exactCtxs, u, ferrors);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD, "Timestep: %04d time = %-8.4g \t L_2 Error: [%2.3g, %2.3g, %2.3g]\n", (int) step, (double) time, (double) ferrors[0], (double) ferrors[1], (double) ferrors[2]);CHKERRQ(ierr);


    PetscFunctionReturn(0);
}

static PetscErrorCode SourceRho(PetscInt dim, PetscReal time, const PetscReal xyz[], PetscInt Nf, PetscScalar *u, void *ctx) {
    PetscFunctionBeginUser;

    Constants *constants = (Constants *)ctx;
    PetscReal L = constants->L;
    PetscReal gamma = constants->gamma;

    PetscReal rhoO = constants->rho.phiO;
    PetscReal rhoX = constants->rho.phiX;
    PetscReal rhoY = constants->rho.phiY;
    PetscReal rhoZ = constants->rho.phiZ;
    PetscReal aRhoX = constants->rho.aPhiX;
    PetscReal aRhoY = constants->rho.aPhiY;
    PetscReal aRhoZ = constants->rho.aPhiZ;

    PetscReal uO = constants->u.phiO;
    PetscReal uX = constants->u.phiX;
    PetscReal uY = constants->u.phiY;
    PetscReal uZ = constants->u.phiZ;
    PetscReal aUX = constants->u.aPhiX;
    PetscReal aUY = constants->u.aPhiY;
    PetscReal aUZ = constants->u.aPhiZ;

    PetscReal vO = constants->v.phiO;
    PetscReal vX = constants->v.phiX;
    PetscReal vY = constants->v.phiY;
    PetscReal vZ = constants->v.phiZ;
    PetscReal aVX = constants->v.aPhiX;
    PetscReal aVY = constants->v.aPhiY;
    PetscReal aVZ = constants->v.aPhiZ;

    PetscReal wO = constants->w.phiO;
    PetscReal wX = constants->w.phiX;
    PetscReal wY = constants->w.phiY;
    PetscReal wZ = constants->w.phiZ;
    PetscReal aWX = constants->w.aPhiX;
    PetscReal aWY = constants->w.aPhiY;
    PetscReal aWZ = constants->w.aPhiZ;

    PetscReal pO = constants->p.phiO;
    PetscReal pX = constants->p.phiX;
    PetscReal pY = constants->p.phiY;
    PetscReal pZ = constants->p.phiZ;
    PetscReal aPX = constants->p.aPhiX;
    PetscReal aPY = constants->p.aPhiY;
    PetscReal aPZ = constants->p.aPhiZ;

    PetscReal x = xyz[0];
    PetscReal y = xyz[1];
    PetscReal z = dim > 2? xyz[2] : 0.0;

    u[0] = (aRhoX*Pi*rhoX*Cos((aRhoX*Pi*x)/L)*(uO + uY*Cos((aUY*Pi*y)/L) + uZ*Cos((aUZ*Pi*z)/L) +
                                               uX*Sin((aUX*Pi*x)/L)))/L + (aRhoZ*Pi*rhoZ*Cos((aRhoZ*Pi*z)/L)*
                                                                           (wO + wZ*Cos((aWZ*Pi*z)/L) + wX*Sin((aWX*Pi*x)/L) + wY*Sin((aWY*Pi*y)/L)))/L +
           (aUX*Pi*uX*Cos((aUX*Pi*x)/L)*(rhoO + rhoY*Cos((aRhoY*Pi*y)/L) + rhoX*Sin((aRhoX*Pi*x)/L) +
                                         rhoZ*Sin((aRhoZ*Pi*z)/L)))/L + (aVY*Pi*vY*Cos((aVY*Pi*y)/L)*
                                                                         (rhoO + rhoY*Cos((aRhoY*Pi*y)/L) + rhoX*Sin((aRhoX*Pi*x)/L) + rhoZ*Sin((aRhoZ*Pi*z)/L)))/L -
           (aRhoY*Pi*rhoY*Sin((aRhoY*Pi*y)/L)*(vO + vX*Cos((aVX*Pi*x)/L) + vY*Sin((aVY*Pi*y)/L) +
                                               vZ*Sin((aVZ*Pi*z)/L)))/L - (aWZ*Pi*wZ*
                                                                           (rhoO + rhoY*Cos((aRhoY*Pi*y)/L) + rhoX*Sin((aRhoX*Pi*x)/L) + rhoZ*Sin((aRhoZ*Pi*z)/L))*
                                                                           Sin((aWZ*Pi*z)/L))/L;
    PetscFunctionReturn(0);
}

static PetscErrorCode SourceRhoU(PetscInt dim, PetscReal time, const PetscReal xyz[], PetscInt Nf, PetscScalar *u, void *ctx) {
    PetscFunctionBeginUser;

    Constants *constants = (Constants *)ctx;
    PetscReal L = constants->L;
    PetscReal gamma = constants->gamma;

    PetscReal rhoO = constants->rho.phiO;
    PetscReal rhoX = constants->rho.phiX;
    PetscReal rhoY = constants->rho.phiY;
    PetscReal rhoZ = constants->rho.phiZ;
    PetscReal aRhoX = constants->rho.aPhiX;
    PetscReal aRhoY = constants->rho.aPhiY;
    PetscReal aRhoZ = constants->rho.aPhiZ;

    PetscReal uO = constants->u.phiO;
    PetscReal uX = constants->u.phiX;
    PetscReal uY = constants->u.phiY;
    PetscReal uZ = constants->u.phiZ;
    PetscReal aUX = constants->u.aPhiX;
    PetscReal aUY = constants->u.aPhiY;
    PetscReal aUZ = constants->u.aPhiZ;

    PetscReal vO = constants->v.phiO;
    PetscReal vX = constants->v.phiX;
    PetscReal vY = constants->v.phiY;
    PetscReal vZ = constants->v.phiZ;
    PetscReal aVX = constants->v.aPhiX;
    PetscReal aVY = constants->v.aPhiY;
    PetscReal aVZ = constants->v.aPhiZ;

    PetscReal wO = constants->w.phiO;
    PetscReal wX = constants->w.phiX;
    PetscReal wY = constants->w.phiY;
    PetscReal wZ = constants->w.phiZ;
    PetscReal aWX = constants->w.aPhiX;
    PetscReal aWY = constants->w.aPhiY;
    PetscReal aWZ = constants->w.aPhiZ;

    PetscReal pO = constants->p.phiO;
    PetscReal pX = constants->p.phiX;
    PetscReal pY = constants->p.phiY;
    PetscReal pZ = constants->p.phiZ;
    PetscReal aPX = constants->p.aPhiX;
    PetscReal aPY = constants->p.aPhiY;
    PetscReal aPZ = constants->p.aPhiZ;

    PetscReal x = xyz[0];
    PetscReal y = xyz[1];
    PetscReal z = dim > 2? xyz[2] : 0.0;


    u[0] = -((aPX*Pi*pX*Sin((aPX*Pi*x)/L))/L) + (aRhoX*Pi*rhoX*Cos((aRhoX*Pi*x)/L)*
                                                 Power(uO + uY*Cos((aUY*Pi*y)/L) + uZ*Cos((aUZ*Pi*z)/L) + uX*Sin((aUX*Pi*x)/L),2))/L +
           (aRhoZ*Pi*rhoZ*Cos((aRhoZ*Pi*z)/L)*(uO + uY*Cos((aUY*Pi*y)/L) + uZ*Cos((aUZ*Pi*z)/L) +
                                               uX*Sin((aUX*Pi*x)/L))*(wO + wZ*Cos((aWZ*Pi*z)/L) + wX*Sin((aWX*Pi*x)/L) + wY*Sin((aWY*Pi*y)/L)))/
           L + (2*aUX*Pi*uX*Cos((aUX*Pi*x)/L)*(uO + uY*Cos((aUY*Pi*y)/L) + uZ*Cos((aUZ*Pi*z)/L) +
                                               uX*Sin((aUX*Pi*x)/L))*(rhoO + rhoY*Cos((aRhoY*Pi*y)/L) + rhoX*Sin((aRhoX*Pi*x)/L) +
                                                                      rhoZ*Sin((aRhoZ*Pi*z)/L)))/L + (aVY*Pi*vY*Cos((aVY*Pi*y)/L)*
                                                                                                      (uO + uY*Cos((aUY*Pi*y)/L) + uZ*Cos((aUZ*Pi*z)/L) + uX*Sin((aUX*Pi*x)/L))*
                                                                                                      (rhoO + rhoY*Cos((aRhoY*Pi*y)/L) + rhoX*Sin((aRhoX*Pi*x)/L) + rhoZ*Sin((aRhoZ*Pi*z)/L)))/L -
           (aUZ*Pi*uZ*(wO + wZ*Cos((aWZ*Pi*z)/L) + wX*Sin((aWX*Pi*x)/L) + wY*Sin((aWY*Pi*y)/L))*
            (rhoO + rhoY*Cos((aRhoY*Pi*y)/L) + rhoX*Sin((aRhoX*Pi*x)/L) + rhoZ*Sin((aRhoZ*Pi*z)/L))*
            Sin((aUZ*Pi*z)/L))/L - (aRhoY*Pi*rhoY*
                                    (uO + uY*Cos((aUY*Pi*y)/L) + uZ*Cos((aUZ*Pi*z)/L) + uX*Sin((aUX*Pi*x)/L))*Sin((aRhoY*Pi*y)/L)*
                                    (vO + vX*Cos((aVX*Pi*x)/L) + vY*Sin((aVY*Pi*y)/L) + vZ*Sin((aVZ*Pi*z)/L)))/L -
           (aUY*Pi*uY*Sin((aUY*Pi*y)/L)*(rhoO + rhoY*Cos((aRhoY*Pi*y)/L) + rhoX*Sin((aRhoX*Pi*x)/L) +
                                         rhoZ*Sin((aRhoZ*Pi*z)/L))*(vO + vX*Cos((aVX*Pi*x)/L) + vY*Sin((aVY*Pi*y)/L) +
                                                                    vZ*Sin((aVZ*Pi*z)/L)))/L - (aWZ*Pi*wZ*
                                                                                                (uO + uY*Cos((aUY*Pi*y)/L) + uZ*Cos((aUZ*Pi*z)/L) + uX*Sin((aUX*Pi*x)/L))*
                                                                                                (rhoO + rhoY*Cos((aRhoY*Pi*y)/L) + rhoX*Sin((aRhoX*Pi*x)/L) + rhoZ*Sin((aRhoZ*Pi*z)/L))*
                                                                                                Sin((aWZ*Pi*z)/L))/L;

    u[1] = (aPY*Pi*pY*Cos((aPY*Pi*y)/L))/L - (aVX*Pi*vX*
                                              (uO + uY*Cos((aUY*Pi*y)/L) + uZ*Cos((aUZ*Pi*z)/L) + uX*Sin((aUX*Pi*x)/L))*Sin((aVX*Pi*x)/L)*
                                              (rhoO + rhoY*Cos((aRhoY*Pi*y)/L) + rhoX*Sin((aRhoX*Pi*x)/L) + rhoZ*Sin((aRhoZ*Pi*z)/L)))/L +
           (aVZ*Pi*vZ*Cos((aVZ*Pi*z)/L)*(wO + wZ*Cos((aWZ*Pi*z)/L) + wX*Sin((aWX*Pi*x)/L) +
                                         wY*Sin((aWY*Pi*y)/L))*(rhoO + rhoY*Cos((aRhoY*Pi*y)/L) + rhoX*Sin((aRhoX*Pi*x)/L) +
                                                                rhoZ*Sin((aRhoZ*Pi*z)/L)))/L + (aRhoX*Pi*rhoX*Cos((aRhoX*Pi*x)/L)*
                                                                                                (uO + uY*Cos((aUY*Pi*y)/L) + uZ*Cos((aUZ*Pi*z)/L) + uX*Sin((aUX*Pi*x)/L))*
                                                                                                (vO + vX*Cos((aVX*Pi*x)/L) + vY*Sin((aVY*Pi*y)/L) + vZ*Sin((aVZ*Pi*z)/L)))/L +
           (aRhoZ*Pi*rhoZ*Cos((aRhoZ*Pi*z)/L)*(wO + wZ*Cos((aWZ*Pi*z)/L) + wX*Sin((aWX*Pi*x)/L) +
                                               wY*Sin((aWY*Pi*y)/L))*(vO + vX*Cos((aVX*Pi*x)/L) + vY*Sin((aVY*Pi*y)/L) + vZ*Sin((aVZ*Pi*z)/L)))/
           L + (aUX*Pi*uX*Cos((aUX*Pi*x)/L)*(rhoO + rhoY*Cos((aRhoY*Pi*y)/L) + rhoX*Sin((aRhoX*Pi*x)/L) +
                                             rhoZ*Sin((aRhoZ*Pi*z)/L))*(vO + vX*Cos((aVX*Pi*x)/L) + vY*Sin((aVY*Pi*y)/L) +
                                                                        vZ*Sin((aVZ*Pi*z)/L)))/L + (2*aVY*Pi*vY*Cos((aVY*Pi*y)/L)*
                                                                                                    (rhoO + rhoY*Cos((aRhoY*Pi*y)/L) + rhoX*Sin((aRhoX*Pi*x)/L) + rhoZ*Sin((aRhoZ*Pi*z)/L))*
                                                                                                    (vO + vX*Cos((aVX*Pi*x)/L) + vY*Sin((aVY*Pi*y)/L) + vZ*Sin((aVZ*Pi*z)/L)))/L -
           (aRhoY*Pi*rhoY*Sin((aRhoY*Pi*y)/L)*Power(vO + vX*Cos((aVX*Pi*x)/L) + vY*Sin((aVY*Pi*y)/L) +
                                                    vZ*Sin((aVZ*Pi*z)/L),2))/L - (aWZ*Pi*wZ*
                                                                                  (rhoO + rhoY*Cos((aRhoY*Pi*y)/L) + rhoX*Sin((aRhoX*Pi*x)/L) + rhoZ*Sin((aRhoZ*Pi*z)/L))*
                                                                                  (vO + vX*Cos((aVX*Pi*x)/L) + vY*Sin((aVY*Pi*y)/L) + vZ*Sin((aVZ*Pi*z)/L))*Sin((aWZ*Pi*z)/L))/L;

    if(dim >2){
        u[2] = (aRhoX*Pi*rhoX*Cos((aRhoX*Pi*x)/L)*(uO + uY*Cos((aUY*Pi*y)/L) + uZ*Cos((aUZ*Pi*z)/L) +
                                                   uX*Sin((aUX*Pi*x)/L))*(wO + wZ*Cos((aWZ*Pi*z)/L) + wX*Sin((aWX*Pi*x)/L) + wY*Sin((aWY*Pi*y)/L)))/
               L + (aRhoZ*Pi*rhoZ*Cos((aRhoZ*Pi*z)/L)*
                    Power(wO + wZ*Cos((aWZ*Pi*z)/L) + wX*Sin((aWX*Pi*x)/L) + wY*Sin((aWY*Pi*y)/L),2))/L -
               (aPZ*Pi*pZ*Sin((aPZ*Pi*z)/L))/L + (aWX*Pi*wX*Cos((aWX*Pi*x)/L)*
                                                  (uO + uY*Cos((aUY*Pi*y)/L) + uZ*Cos((aUZ*Pi*z)/L) + uX*Sin((aUX*Pi*x)/L))*
                                                  (rhoO + rhoY*Cos((aRhoY*Pi*y)/L) + rhoX*Sin((aRhoX*Pi*x)/L) + rhoZ*Sin((aRhoZ*Pi*z)/L)))/L +
               (aUX*Pi*uX*Cos((aUX*Pi*x)/L)*(wO + wZ*Cos((aWZ*Pi*z)/L) + wX*Sin((aWX*Pi*x)/L) +
                                             wY*Sin((aWY*Pi*y)/L))*(rhoO + rhoY*Cos((aRhoY*Pi*y)/L) + rhoX*Sin((aRhoX*Pi*x)/L) +
                                                                    rhoZ*Sin((aRhoZ*Pi*z)/L)))/L + (aVY*Pi*vY*Cos((aVY*Pi*y)/L)*
                                                                                                    (wO + wZ*Cos((aWZ*Pi*z)/L) + wX*Sin((aWX*Pi*x)/L) + wY*Sin((aWY*Pi*y)/L))*
                                                                                                    (rhoO + rhoY*Cos((aRhoY*Pi*y)/L) + rhoX*Sin((aRhoX*Pi*x)/L) + rhoZ*Sin((aRhoZ*Pi*z)/L)))/L -
               (aRhoY*Pi*rhoY*Sin((aRhoY*Pi*y)/L)*(wO + wZ*Cos((aWZ*Pi*z)/L) + wX*Sin((aWX*Pi*x)/L) +
                                                   wY*Sin((aWY*Pi*y)/L))*(vO + vX*Cos((aVX*Pi*x)/L) + vY*Sin((aVY*Pi*y)/L) + vZ*Sin((aVZ*Pi*z)/L)))/
               L + (aWY*Pi*wY*Cos((aWY*Pi*y)/L)*(rhoO + rhoY*Cos((aRhoY*Pi*y)/L) + rhoX*Sin((aRhoX*Pi*x)/L) +
                                                 rhoZ*Sin((aRhoZ*Pi*z)/L))*(vO + vX*Cos((aVX*Pi*x)/L) + vY*Sin((aVY*Pi*y)/L) +
                                                                            vZ*Sin((aVZ*Pi*z)/L)))/L - (2*aWZ*Pi*wZ*
                                                                                                        (wO + wZ*Cos((aWZ*Pi*z)/L) + wX*Sin((aWX*Pi*x)/L) + wY*Sin((aWY*Pi*y)/L))*
                                                                                                        (rhoO + rhoY*Cos((aRhoY*Pi*y)/L) + rhoX*Sin((aRhoX*Pi*x)/L) + rhoZ*Sin((aRhoZ*Pi*z)/L))*
                                                                                                        Sin((aWZ*Pi*z)/L))/L;
    }
    PetscFunctionReturn(0);
}

static PetscErrorCode SourceRhoE(PetscInt dim, PetscReal time, const PetscReal xyz[], PetscInt Nf, PetscScalar *u, void *ctx) {
    PetscFunctionBeginUser;

    Constants *constants = (Constants *)ctx;
    PetscReal L = constants->L;
    PetscReal gamma = constants->gamma;

    PetscReal rhoO = constants->rho.phiO;
    PetscReal rhoX = constants->rho.phiX;
    PetscReal rhoY = constants->rho.phiY;
    PetscReal rhoZ = constants->rho.phiZ;
    PetscReal aRhoX = constants->rho.aPhiX;
    PetscReal aRhoY = constants->rho.aPhiY;
    PetscReal aRhoZ = constants->rho.aPhiZ;

    PetscReal uO = constants->u.phiO;
    PetscReal uX = constants->u.phiX;
    PetscReal uY = constants->u.phiY;
    PetscReal uZ = constants->u.phiZ;
    PetscReal aUX = constants->u.aPhiX;
    PetscReal aUY = constants->u.aPhiY;
    PetscReal aUZ = constants->u.aPhiZ;

    PetscReal vO = constants->v.phiO;
    PetscReal vX = constants->v.phiX;
    PetscReal vY = constants->v.phiY;
    PetscReal vZ = constants->v.phiZ;
    PetscReal aVX = constants->v.aPhiX;
    PetscReal aVY = constants->v.aPhiY;
    PetscReal aVZ = constants->v.aPhiZ;

    PetscReal wO = constants->w.phiO;
    PetscReal wX = constants->w.phiX;
    PetscReal wY = constants->w.phiY;
    PetscReal wZ = constants->w.phiZ;
    PetscReal aWX = constants->w.aPhiX;
    PetscReal aWY = constants->w.aPhiY;
    PetscReal aWZ = constants->w.aPhiZ;

    PetscReal pO = constants->p.phiO;
    PetscReal pX = constants->p.phiX;
    PetscReal pY = constants->p.phiY;
    PetscReal pZ = constants->p.phiZ;
    PetscReal aPX = constants->p.aPhiX;
    PetscReal aPY = constants->p.aPhiY;
    PetscReal aPZ = constants->p.aPhiZ;

    PetscReal x = xyz[0];
    PetscReal y = xyz[1];
    PetscReal z = dim > 2? xyz[2] : 0.0;

    u[0] = -((aPX*Pi*pX*Sin((aPX*Pi*x)/L)*(uO + uY*Cos((aUY*Pi*y)/L) + uZ*Cos((aUZ*Pi*z)/L) + uX*Sin((aUX*Pi*x)/L)))/L) + (aUX*Pi*uX*Cos((aUX*Pi*x)/L)*(pO + pX*Cos((aPX*Pi*x)/L) + pZ*Cos((aPZ*Pi*z)/L) + pY*Sin((aPY*Pi*y)/L)))/L +
           (aVY*Pi*vY*Cos((aVY*Pi*y)/L)*(pO + pX*Cos((aPX*Pi*x)/L) + pZ*Cos((aPZ*Pi*z)/L) + pY*Sin((aPY*Pi*y)/L)))/L - (aPZ*Pi*pZ*(wO + wZ*Cos((aWZ*Pi*z)/L) + wX*Sin((aWX*Pi*x)/L) + wY*Sin((aWY*Pi*y)/L))*Sin((aPZ*Pi*z)/L))/L +
           (aPY*Pi*pY*Cos((aPY*Pi*y)/L)*(vO + vX*Cos((aVX*Pi*x)/L) + vY*Sin((aVY*Pi*y)/L) + vZ*Sin((aVZ*Pi*z)/L)))/L + (rhoO + rhoY*Cos((aRhoY*Pi*y)/L) + rhoX*Sin((aRhoX*Pi*x)/L) + rhoZ*Sin((aRhoZ*Pi*z)/L))*(vO + vX*Cos((aVX*Pi*x)/L) + vY*Sin((aVY*Pi*y)/L) + vZ*Sin((aVZ*Pi*z)/L))*
                                                                                                                       ((aRhoY*Pi*rhoY*(pO + pX*Cos((aPX*Pi*x)/L) + pZ*Cos((aPZ*Pi*z)/L) + pY*Sin((aPY*Pi*y)/L))*Sin((aRhoY*Pi*y)/L))/((-1. + gamma)*L*Power(rhoO + rhoY*Cos((aRhoY*Pi*y)/L) + rhoX*Sin((aRhoX*Pi*x)/L) + rhoZ*Sin((aRhoZ*Pi*z)/L),2)) +
                                                                                                                        (aPY*Pi*pY*Cos((aPY*Pi*y)/L))/((-1. + gamma)*L*(rhoO + rhoY*Cos((aRhoY*Pi*y)/L) + rhoX*Sin((aRhoX*Pi*x)/L) + rhoZ*Sin((aRhoZ*Pi*z)/L))) +
                                                                                                                        ((-2*aUY*Pi*uY*(uO + uY*Cos((aUY*Pi*y)/L) + uZ*Cos((aUZ*Pi*z)/L) + uX*Sin((aUX*Pi*x)/L))*Sin((aUY*Pi*y)/L))/L + (2*aWY*Pi*wY*Cos((aWY*Pi*y)/L)*(wO + wZ*Cos((aWZ*Pi*z)/L) + wX*Sin((aWX*Pi*x)/L) + wY*Sin((aWY*Pi*y)/L)))/L +
                                                                                                                         (2*aVY*Pi*vY*Cos((aVY*Pi*y)/L)*(vO + vX*Cos((aVX*Pi*x)/L) + vY*Sin((aVY*Pi*y)/L) + vZ*Sin((aVZ*Pi*z)/L)))/L)/2.) + (uO + uY*Cos((aUY*Pi*y)/L) + uZ*Cos((aUZ*Pi*z)/L) + uX*Sin((aUX*Pi*x)/L))*(rhoO + rhoY*Cos((aRhoY*Pi*y)/L) + rhoX*Sin((aRhoX*Pi*x)/L) + rhoZ*Sin((aRhoZ*Pi*z)/L))*
                                                                                                                                                                                                                                            (-((aRhoX*Pi*rhoX*Cos((aRhoX*Pi*x)/L)*(pO + pX*Cos((aPX*Pi*x)/L) + pZ*Cos((aPZ*Pi*z)/L) + pY*Sin((aPY*Pi*y)/L)))/((-1. + gamma)*L*Power(rhoO + rhoY*Cos((aRhoY*Pi*y)/L) + rhoX*Sin((aRhoX*Pi*x)/L) + rhoZ*Sin((aRhoZ*Pi*z)/L),2))) -
                                                                                                                                                                                                                                             (aPX*Pi*pX*Sin((aPX*Pi*x)/L))/((-1. + gamma)*L*(rhoO + rhoY*Cos((aRhoY*Pi*y)/L) + rhoX*Sin((aRhoX*Pi*x)/L) + rhoZ*Sin((aRhoZ*Pi*z)/L))) +
                                                                                                                                                                                                                                             ((2*aUX*Pi*uX*Cos((aUX*Pi*x)/L)*(uO + uY*Cos((aUY*Pi*y)/L) + uZ*Cos((aUZ*Pi*z)/L) + uX*Sin((aUX*Pi*x)/L)))/L + (2*aWX*Pi*wX*Cos((aWX*Pi*x)/L)*(wO + wZ*Cos((aWZ*Pi*z)/L) + wX*Sin((aWX*Pi*x)/L) + wY*Sin((aWY*Pi*y)/L)))/L -
                                                                                                                                                                                                                                              (2*aVX*Pi*vX*Sin((aVX*Pi*x)/L)*(vO + vX*Cos((aVX*Pi*x)/L) + vY*Sin((aVY*Pi*y)/L) + vZ*Sin((aVZ*Pi*z)/L)))/L)/2.) + (aRhoX*Pi*rhoX*Cos((aRhoX*Pi*x)/L)*(uO + uY*Cos((aUY*Pi*y)/L) + uZ*Cos((aUZ*Pi*z)/L) + uX*Sin((aUX*Pi*x)/L))*
                                                                                                                                                                                                                                                                                                                                                                  ((pO + pX*Cos((aPX*Pi*x)/L) + pZ*Cos((aPZ*Pi*z)/L) + pY*Sin((aPY*Pi*y)/L))/((-1. + gamma)*(rhoO + rhoY*Cos((aRhoY*Pi*y)/L) + rhoX*Sin((aRhoX*Pi*x)/L) + rhoZ*Sin((aRhoZ*Pi*z)/L))) +
                                                                                                                                                                                                                                                                                                                                                                   (Power(uO + uY*Cos((aUY*Pi*y)/L) + uZ*Cos((aUZ*Pi*z)/L) + uX*Sin((aUX*Pi*x)/L),2) + Power(wO + wZ*Cos((aWZ*Pi*z)/L) + wX*Sin((aWX*Pi*x)/L) + wY*Sin((aWY*Pi*y)/L),2) + Power(vO + vX*Cos((aVX*Pi*x)/L) + vY*Sin((aVY*Pi*y)/L) + vZ*Sin((aVZ*Pi*z)/L),2))/2.))/L +
           (aRhoZ*Pi*rhoZ*Cos((aRhoZ*Pi*z)/L)*(wO + wZ*Cos((aWZ*Pi*z)/L) + wX*Sin((aWX*Pi*x)/L) + wY*Sin((aWY*Pi*y)/L))*((pO + pX*Cos((aPX*Pi*x)/L) + pZ*Cos((aPZ*Pi*z)/L) + pY*Sin((aPY*Pi*y)/L))/((-1. + gamma)*(rhoO + rhoY*Cos((aRhoY*Pi*y)/L) + rhoX*Sin((aRhoX*Pi*x)/L) + rhoZ*Sin((aRhoZ*Pi*z)/L))) +
                                                                                                                         (Power(uO + uY*Cos((aUY*Pi*y)/L) + uZ*Cos((aUZ*Pi*z)/L) + uX*Sin((aUX*Pi*x)/L),2) + Power(wO + wZ*Cos((aWZ*Pi*z)/L) + wX*Sin((aWX*Pi*x)/L) + wY*Sin((aWY*Pi*y)/L),2) + Power(vO + vX*Cos((aVX*Pi*x)/L) + vY*Sin((aVY*Pi*y)/L) + vZ*Sin((aVZ*Pi*z)/L),2))/2.))/L +
           (aUX*Pi*uX*Cos((aUX*Pi*x)/L)*(rhoO + rhoY*Cos((aRhoY*Pi*y)/L) + rhoX*Sin((aRhoX*Pi*x)/L) + rhoZ*Sin((aRhoZ*Pi*z)/L))*((pO + pX*Cos((aPX*Pi*x)/L) + pZ*Cos((aPZ*Pi*z)/L) + pY*Sin((aPY*Pi*y)/L))/((-1. + gamma)*(rhoO + rhoY*Cos((aRhoY*Pi*y)/L) + rhoX*Sin((aRhoX*Pi*x)/L) + rhoZ*Sin((aRhoZ*Pi*z)/L))) +
                                                                                                                                 (Power(uO + uY*Cos((aUY*Pi*y)/L) + uZ*Cos((aUZ*Pi*z)/L) + uX*Sin((aUX*Pi*x)/L),2) + Power(wO + wZ*Cos((aWZ*Pi*z)/L) + wX*Sin((aWX*Pi*x)/L) + wY*Sin((aWY*Pi*y)/L),2) + Power(vO + vX*Cos((aVX*Pi*x)/L) + vY*Sin((aVY*Pi*y)/L) + vZ*Sin((aVZ*Pi*z)/L),2))/2.))/L +
           (aVY*Pi*vY*Cos((aVY*Pi*y)/L)*(rhoO + rhoY*Cos((aRhoY*Pi*y)/L) + rhoX*Sin((aRhoX*Pi*x)/L) + rhoZ*Sin((aRhoZ*Pi*z)/L))*((pO + pX*Cos((aPX*Pi*x)/L) + pZ*Cos((aPZ*Pi*z)/L) + pY*Sin((aPY*Pi*y)/L))/((-1. + gamma)*(rhoO + rhoY*Cos((aRhoY*Pi*y)/L) + rhoX*Sin((aRhoX*Pi*x)/L) + rhoZ*Sin((aRhoZ*Pi*z)/L))) +
                                                                                                                                 (Power(uO + uY*Cos((aUY*Pi*y)/L) + uZ*Cos((aUZ*Pi*z)/L) + uX*Sin((aUX*Pi*x)/L),2) + Power(wO + wZ*Cos((aWZ*Pi*z)/L) + wX*Sin((aWX*Pi*x)/L) + wY*Sin((aWY*Pi*y)/L),2) + Power(vO + vX*Cos((aVX*Pi*x)/L) + vY*Sin((aVY*Pi*y)/L) + vZ*Sin((aVZ*Pi*z)/L),2))/2.))/L -
           (aRhoY*Pi*rhoY*Sin((aRhoY*Pi*y)/L)*(vO + vX*Cos((aVX*Pi*x)/L) + vY*Sin((aVY*Pi*y)/L) + vZ*Sin((aVZ*Pi*z)/L))*((pO + pX*Cos((aPX*Pi*x)/L) + pZ*Cos((aPZ*Pi*z)/L) + pY*Sin((aPY*Pi*y)/L))/((-1. + gamma)*(rhoO + rhoY*Cos((aRhoY*Pi*y)/L) + rhoX*Sin((aRhoX*Pi*x)/L) + rhoZ*Sin((aRhoZ*Pi*z)/L))) +
                                                                                                                         (Power(uO + uY*Cos((aUY*Pi*y)/L) + uZ*Cos((aUZ*Pi*z)/L) + uX*Sin((aUX*Pi*x)/L),2) + Power(wO + wZ*Cos((aWZ*Pi*z)/L) + wX*Sin((aWX*Pi*x)/L) + wY*Sin((aWY*Pi*y)/L),2) + Power(vO + vX*Cos((aVX*Pi*x)/L) + vY*Sin((aVY*Pi*y)/L) + vZ*Sin((aVZ*Pi*z)/L),2))/2.))/L -
           (aWZ*Pi*wZ*(pO + pX*Cos((aPX*Pi*x)/L) + pZ*Cos((aPZ*Pi*z)/L) + pY*Sin((aPY*Pi*y)/L))*Sin((aWZ*Pi*z)/L))/L - (aWZ*Pi*wZ*(rhoO + rhoY*Cos((aRhoY*Pi*y)/L) + rhoX*Sin((aRhoX*Pi*x)/L) + rhoZ*Sin((aRhoZ*Pi*z)/L))*
                                                                                                                        ((pO + pX*Cos((aPX*Pi*x)/L) + pZ*Cos((aPZ*Pi*z)/L) + pY*Sin((aPY*Pi*y)/L))/((-1. + gamma)*(rhoO + rhoY*Cos((aRhoY*Pi*y)/L) + rhoX*Sin((aRhoX*Pi*x)/L) + rhoZ*Sin((aRhoZ*Pi*z)/L))) +
                                                                                                                         (Power(uO + uY*Cos((aUY*Pi*y)/L) + uZ*Cos((aUZ*Pi*z)/L) + uX*Sin((aUX*Pi*x)/L),2) + Power(wO + wZ*Cos((aWZ*Pi*z)/L) + wX*Sin((aWX*Pi*x)/L) + wY*Sin((aWY*Pi*y)/L),2) + Power(vO + vX*Cos((aVX*Pi*x)/L) + vY*Sin((aVY*Pi*y)/L) + vZ*Sin((aVZ*Pi*z)/L),2))/2.)*Sin((aWZ*Pi*z)/L))/L +
           (wO + wZ*Cos((aWZ*Pi*z)/L) + wX*Sin((aWX*Pi*x)/L) + wY*Sin((aWY*Pi*y)/L))*(rhoO + rhoY*Cos((aRhoY*Pi*y)/L) + rhoX*Sin((aRhoX*Pi*x)/L) + rhoZ*Sin((aRhoZ*Pi*z)/L))*
           (-((aRhoZ*Pi*rhoZ*Cos((aRhoZ*Pi*z)/L)*(pO + pX*Cos((aPX*Pi*x)/L) + pZ*Cos((aPZ*Pi*z)/L) + pY*Sin((aPY*Pi*y)/L)))/((-1. + gamma)*L*Power(rhoO + rhoY*Cos((aRhoY*Pi*y)/L) + rhoX*Sin((aRhoX*Pi*x)/L) + rhoZ*Sin((aRhoZ*Pi*z)/L),2))) -
            (aPZ*Pi*pZ*Sin((aPZ*Pi*z)/L))/((-1. + gamma)*L*(rhoO + rhoY*Cos((aRhoY*Pi*y)/L) + rhoX*Sin((aRhoX*Pi*x)/L) + rhoZ*Sin((aRhoZ*Pi*z)/L))) +
            ((-2*aUZ*Pi*uZ*(uO + uY*Cos((aUY*Pi*y)/L) + uZ*Cos((aUZ*Pi*z)/L) + uX*Sin((aUX*Pi*x)/L))*Sin((aUZ*Pi*z)/L))/L + (2*aVZ*Pi*vZ*Cos((aVZ*Pi*z)/L)*(vO + vX*Cos((aVX*Pi*x)/L) + vY*Sin((aVY*Pi*y)/L) + vZ*Sin((aVZ*Pi*z)/L)))/L -
             (2*aWZ*Pi*wZ*(wO + wZ*Cos((aWZ*Pi*z)/L) + wX*Sin((aWX*Pi*x)/L) + wY*Sin((aWY*Pi*y)/L))*Sin((aWZ*Pi*z)/L))/L)/2.);
    PetscFunctionReturn(0);
}


static PetscErrorCode PhysicsBoundary_Euler(PetscReal time, const PetscReal *c, const PetscReal *n, const PetscScalar *a_xI, PetscScalar *a_xG, void *ctx) {
    PetscFunctionBeginUser;
    Constants *constants = (Constants *)ctx;

    RhoExact(constants->dim, time, c, 0, a_xG + RHO, ctx);
    RhoEExact(constants->dim, time, c, 0, a_xG + RHOE, ctx);
    RhoUExact(constants->dim, time, c, 0, a_xG + RHOU, ctx);

    PetscFunctionReturn(0);
}

static PetscErrorCode ComputeRHSWithSourceTerms(DM dm, PetscReal time, Vec locXVec, Vec globFVec, void *ctx){
    PetscFunctionBeginUser;
    PetscErrorCode ierr;
    ProblemSetup *setup = (ProblemSetup *)ctx;

    // Call the flux calculation
    ierr = DMPlexTSComputeRHSFunctionFVM(dm, time, locXVec, globFVec, setup->flowData);CHKERRQ(ierr);

    // Convert the dm to a plex
    DM plex;
    DMConvert(dm, DMPLEX, &plex);

    // Extract the cell geometry, and the dm that holds the information
    Vec cellgeom;
    DM dmCell;
    const PetscScalar *cgeom;
    ierr = DMPlexGetGeometryFVM(plex, NULL, &cellgeom, NULL);CHKERRQ(ierr);
    ierr = VecGetDM(cellgeom, &dmCell);CHKERRQ(ierr);
    ierr = VecGetArrayRead(cellgeom, &cgeom);CHKERRQ(ierr);

    // Get the cell start and end for the fv cells
    PetscInt cStart, cEnd;
    ierr = DMPlexGetSimplexOrBoxCells(dmCell, 0, &cStart, &cEnd);CHKERRQ(ierr);

    // create a local f vector
    Vec locFVec;
    PetscScalar  *locFArray;
    ierr = DMGetLocalVector(dm, &locFVec);CHKERRQ(ierr);
    ierr = VecZeroEntries(locFVec);CHKERRQ(ierr);
    ierr = VecGetArray(locFVec, &locFArray);CHKERRQ(ierr);

    // get the current values
    const PetscScalar      *locXArray;
    ierr = VecGetArrayRead(locXVec, &locXArray);CHKERRQ(ierr);

    PetscInt rank;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

    // March over each cell volume
    for (PetscInt c = cStart; c < cEnd; ++c) {
        PetscFVCellGeom       *cg;
        const PetscReal           *xc;
        PetscReal           *fc;

        ierr = DMPlexPointLocalRead(dmCell, c, cgeom, &cg);CHKERRQ(ierr);
        ierr = DMPlexPointLocalFieldRead(plex, c, 0, locXArray, &xc);CHKERRQ(ierr);
        ierr = DMPlexPointGlobalFieldRef(plex, c, 0, locFArray, &fc);CHKERRQ(ierr);

        if(fc) {  // must be real cell and not ghost
            SourceRho(setup->constants.dim, time, cg->centroid, 0, fc + RHO, &setup->constants);
            SourceRhoE(setup->constants.dim, time, cg->centroid, 0, fc + RHOE, &setup->constants);
            SourceRhoU(setup->constants.dim, time, cg->centroid, 0, fc + RHOU, &setup->constants);
        }
    }

    // restore the cell geometry
    ierr = VecRestoreArrayRead(cellgeom, &cgeom);CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(locXVec, &locXArray);CHKERRQ(ierr);
    ierr = VecRestoreArray(locFVec, &locFArray);CHKERRQ(ierr);

    ierr = DMLocalToGlobalBegin(dm, locFVec, ADD_VALUES, globFVec);CHKERRQ(ierr);
    ierr = DMLocalToGlobalEnd(dm, locFVec, ADD_VALUES, globFVec);CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(dm, &locFVec);CHKERRQ(ierr);

    {// check rhs,
        // temp read current residual
        const PetscScalar *currentFArray;
        ierr = VecGetArrayRead(globFVec, &currentFArray);CHKERRQ(ierr);
        ierr = VecGetArrayRead(cellgeom, &cgeom);CHKERRQ(ierr);

        // March over each cell volume
        for (PetscInt c = cStart; c < cEnd; ++c) {
            PetscFVCellGeom       *cg;
            const PetscReal           *fcCurrent;

            ierr = DMPlexPointLocalRead(dmCell, c, cgeom, &cg);CHKERRQ(ierr);
            ierr = DMPlexPointGlobalFieldRead(plex, c, 0, currentFArray, &fcCurrent);CHKERRQ(ierr);

            if(fcCurrent) {  // must be real cell and not ghost
                if(PetscAbsReal(cg->centroid[0] - .5 ) < 1E-8 && PetscAbsReal(cg->centroid[1] - .5 )  < 1E-8){
                    printf("Residual(%f, %f): %f %f %f %f\n", cg->centroid[0], cg->centroid[1], fcCurrent[0], fcCurrent[1], fcCurrent[2], fcCurrent[3]);
                }
            }
        }

        // temp return current residual
        ierr = VecRestoreArrayRead(globFVec, &currentFArray);CHKERRQ(ierr);
        ierr = VecRestoreArrayRead(cellgeom, &cgeom);CHKERRQ(ierr);
    }



    PetscFunctionReturn(0);
}

PetscErrorCode DMTSCheckResidual(TS ts, DM dm, PetscReal t, Vec u, Vec u_t, PetscReal tol, PetscReal *residual)
{
    MPI_Comm       comm;
    Vec            r;
    PetscReal      res;
    PetscErrorCode ierr;

    PetscFunctionBeginUser;

    ierr = PetscObjectGetComm((PetscObject) ts, &comm);CHKERRQ(ierr);
    ierr = DMComputeExactSolution(dm, t, u, u_t);CHKERRQ(ierr);
    ierr = VecDuplicate(u, &r);CHKERRQ(ierr);
    ierr = TSComputeIFunction(ts, t, u, u_t, r, PETSC_FALSE);CHKERRQ(ierr);
    ierr = VecNorm(r, NORM_2, &res);CHKERRQ(ierr);
    if (tol >= 0.0) {
        if (res > tol) SETERRQ2(comm, PETSC_ERR_ARG_WRONG, "L_2 Residual %g exceeds tolerance %g", (double) res, (double) tol);
    } else if (residual) {
        *residual = res;
    } else {
        ierr = PetscPrintf(comm, "L_2 Residual: %g\n", (double)res);CHKERRQ(ierr);
        ierr = VecChop(r, 1.0e-10);CHKERRQ(ierr);
        ierr = PetscObjectCompose((PetscObject) r, "__Vec_bc_zero__", (PetscObject) dm);CHKERRQ(ierr);
        ierr = PetscObjectSetName((PetscObject) r, "Initial Residual");CHKERRQ(ierr);
        ierr = PetscObjectSetOptionsPrefix((PetscObject)r,"res_");CHKERRQ(ierr);
        ierr = VecViewFromOptions(r, NULL, "-vec_view");CHKERRQ(ierr);
        ierr = PetscObjectCompose((PetscObject) r, "__Vec_bc_zero__", NULL);CHKERRQ(ierr);
    }
    ierr = VecDestroy(&r);CHKERRQ(ierr);
    PetscFunctionReturn(0);
}


int main(int argc, char **argv)
{

    // Setup the problem
    Constants constants;

    // sub sonic
    constants.dim = 2;
    constants.L = 1.0;
    constants.gamma = 1.4;
    constants.R = 287.0;
    constants.mu = 10;

    constants.rho.phiO = 1.0;
    constants.rho.phiX = 0.15;
    constants.rho.phiY = -0.1;
    constants.rho.phiZ = 0.0;
    constants.rho.aPhiX = 1.0;
    constants.rho.aPhiY = 0.5;
    constants.rho.aPhiZ = 0.0;

    constants.u.phiO = 70;
    constants.u.phiX = 5;
    constants.u.phiY = -7;
    constants.u.phiZ = 0.0;
    constants.u.aPhiX = 1.5;
    constants.u.aPhiY = 0.6;
    constants.u.aPhiZ = 0.0;

    constants.v.phiO = 90;
    constants.v.phiX = -15;
    constants.v.phiY = -8.5;
    constants.v.phiZ = 0.0;
    constants.v.aPhiX = 0.5;
    constants.v.aPhiY = 2.0/3.0;
    constants.v.aPhiZ = 0.0;

    constants.w.phiO = 0.0;
    constants.w.phiX = 0.0;
    constants.w.phiY = 0.0;
    constants.w.phiZ = 0.0;
    constants.w.aPhiX = 0.0;
    constants.w.aPhiY = 0.0;
    constants.w.aPhiZ = 0.0;

    constants.p.phiO = 1E5;
    constants.p.phiX = 0.2E5;
    constants.p.phiY = 0.5E5;
    constants.p.phiZ = 0.0;
    constants.p.aPhiX = 2.0;
    constants.p.aPhiY = 1.0;
    constants.p.aPhiZ = 0.0;


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
    PetscReal end[] = {constants.L, constants.L};
    PetscInt nx[] = {129, 129};
    DMBoundaryType bcType[] = {DM_BOUNDARY_NONE, DM_BOUNDARY_NONE};
    ierr = DMPlexCreateBoxMesh(PETSC_COMM_WORLD, constants.dim, PETSC_FALSE, nx, start, end, bcType, PETSC_TRUE, &dm);CHKERRQ(ierr);

//    // Output the mesh
    ierr = DMViewFromOptions(dm, NULL, "-dm_view");CHKERRQ(ierr);

    // Setup the flow data
    FlowData flowData;     /* store some of the flow data*/
    ierr = FlowCreate(&flowData);CHKERRQ(ierr);

    // Combine the flow data
    ProblemSetup problemSetup;
    problemSetup.flowData = flowData;
    problemSetup.constants = constants;

    //Setup
    CompressibleFlow_SetupDiscretization(flowData, dm);

    // Add in the flow parameters
    PetscScalar params[TOTAL_COMPRESSIBLE_FLOW_PARAMETERS];
    params[CFL] = 0.5;
    params[GAMMA] = constants.gamma;

    // set up the finite volume fluxes
    CompressibleFlow_StartProblemSetup(flowData, TOTAL_COMPRESSIBLE_FLOW_PARAMETERS, params);
    DMView(flowData->dm, PETSC_VIEWER_STDERR_SELF);
    // Add in any boundary conditions
    PetscDS prob;
    ierr = DMGetDS(flowData->dm, &prob);
    CHKERRABORT(PETSC_COMM_WORLD, ierr);
    const PetscInt idsLeft[]= {1, 2, 3, 4};
    ierr = PetscDSAddBoundary(prob, DM_BC_NATURAL_RIEMANN, "wall left", "Face Sets", 0, 0, NULL, (void (*)(void))PhysicsBoundary_Euler, NULL, 4, idsLeft, &constants);CHKERRQ(ierr);

    // Complete the problem setup
    ierr = CompressibleFlow_CompleteProblemSetup(flowData, ts);
    CHKERRABORT(PETSC_COMM_WORLD, ierr);

    // Override the flow calc for now
    ierr = DMTSSetRHSFunctionLocal(flowData->dm, ComputeRHSWithSourceTerms, &problemSetup);CHKERRQ(ierr);

    // Name the flow field
    ierr = PetscObjectSetName(((PetscObject)flowData->flowField), "Numerical Solution");
    CHKERRABORT(PETSC_COMM_WORLD, ierr);

    // Setup the TS
    ierr = TSSetFromOptions(ts);
    CHKERRABORT(PETSC_COMM_WORLD, ierr);
    ierr = TSMonitorSet(ts, MonitorError, &constants, NULL);CHKERRQ(ierr);
    CHKERRABORT(PETSC_COMM_WORLD, ierr);
    ierr = TSSetMaxTime(ts, 0.01);CHKERRQ(ierr);

    // set the initial conditions
    PetscErrorCode     (*func[3]) (PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar *u, void *ctx) = {RhoExact, RhoEExact, RhoUExact};
    void* ctxs[3] ={&constants, &constants, &constants};
    ierr    = DMProjectFunction(flowData->dm,0.0,func,ctxs,INSERT_ALL_VALUES,flowData->flowField);CHKERRQ(ierr);

    // for the mms, add the exact solution
    ierr = PetscDSSetExactSolution(prob, RHO, RhoExact, &constants);CHKERRQ(ierr);
    ierr = PetscDSSetExactSolution(prob, RHOE, RhoEExact, &constants);CHKERRQ(ierr);
    ierr = PetscDSSetExactSolution(prob, RHOU, RhoUExact, &constants);CHKERRQ(ierr);
    ierr = PetscDSSetExactSolutionTimeDerivative(prob, RHO, RhoExactTimeDerivative, &constants);CHKERRQ(ierr);
    ierr = PetscDSSetExactSolutionTimeDerivative(prob, RHOE, RhoEExactTimeDerivative, &constants);CHKERRQ(ierr);
    ierr = PetscDSSetExactSolutionTimeDerivative(prob, RHOU, RhoUExactTimeDerivative, &constants);CHKERRQ(ierr);

    // Output the mesh
    PetscReal time = 0.0;
    {
        Vec sol;
        VecDuplicate(flowData->flowField, &sol);
        VecCopy(flowData->flowField, sol);

        SNES snes;
        ierr = TSGetSNES(ts, &snes);CHKERRQ(ierr);
        ierr = DMSNESCheckDiscretization(snes, flowData->dm, time, sol, -1.0, NULL);CHKERRQ(ierr);

        Vec u_t;
        ierr = DMGetGlobalVector(flowData->dm, &u_t);CHKERRQ(ierr);
        ierr = DMTSCheckResidual(ts, flowData->dm, time, sol, u_t, -1.0, NULL);CHKERRQ(ierr);
        ierr = DMRestoreGlobalVector(flowData->dm, &u_t);CHKERRQ(ierr);

        VecDestroy(&sol);
    }

    TSSetMaxSteps(ts, 1);
    ierr = TSSolve(ts,flowData->flowField);CHKERRQ(ierr);

    {
        Vec sol;
        VecDuplicate(flowData->flowField, &sol);
        VecCopy(flowData->flowField, sol);

        SNES snes;
        ierr = TSGetSNES(ts, &snes);CHKERRQ(ierr);
        ierr = DMSNESCheckDiscretization(snes, flowData->dm, time, sol, -1.0, NULL);CHKERRQ(ierr);

        Vec u_t;
        ierr = DMGetGlobalVector(flowData->dm, &u_t);CHKERRQ(ierr);
        ierr = DMTSCheckResidual(ts, flowData->dm, time, sol, u_t, -1.0, NULL);CHKERRQ(ierr);
        ierr = DMRestoreGlobalVector(flowData->dm, &u_t);CHKERRQ(ierr);

        VecDestroy(&sol);
    }

    return PetscFinalize();

}