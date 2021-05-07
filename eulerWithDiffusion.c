#include <petsc.h>
#include <petscmath.h>
#include "compressibleFlow.h"
#include "mesh.h"
#include "petscdmplex.h"
#include "petscts.h"

typedef struct {
    PetscInt dim;
    PetscReal k;
    PetscReal gamma;
    PetscReal Rgas;
    PetscReal L;
} Constants;

typedef struct {
    Constants constants;
    FlowData flowData;
} ProblemSetup;

static PetscErrorCode InitialConditions(PetscInt dim, PetscReal time, const PetscReal xyz[], PetscInt Nf, PetscScalar *node, void *ctx) {
    PetscFunctionBeginUser;

    Constants *constants = (Constants *)ctx;

    PetscReal u = 0.0;
    PetscReal v = 0.0;
    PetscReal rho = 1.0;

    PetscReal Ti = 100;
    PetscReal Tb = 300;

    PetscReal cp = constants->gamma*constants->Rgas/(constants->gamma - 1);

    PetscReal alpha = constants->k/(rho*cp);

    // https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&ved=2ahUKEwj4hvHBwrXwAhWJY98KHSJnCl8QFjAIegQIBBAD&url=https%3A%2F%2Fwww.researchgate.net%2Ffile.PostFileLoader.html%3Fid%3D56882e7a614325f8078b457c%26assetKey%3DAS%253A313540840230912%25401451765369569&usg=AOvVaw1HgQiQUKKdHeQk_QI8O10r
    // https://ocw.mit.edu/courses/mathematics/18-303-linear-partial-differential-equations-fall-2006/lecture-notes/heateqni.pdf
    PetscReal T = 0.0;
    for(PetscInt n =1; n < 10000; n ++){
        PetscReal Bn = -Ti * 2.0*(-1.0 + PetscPowRealInt(-1.0, n))/(PETSC_PI*n);
        PetscReal omegaN = n*PETSC_PI/constants->L;
        T += Bn * PetscSinReal(omegaN * xyz[0])* PetscExpReal(- omegaN*omegaN*alpha*time);
    }
    T += Tb;

    PetscReal p= rho*constants->Rgas*T;
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
    PetscFunctionReturn(0);
}

static PetscReal computeTemperature(PetscInt dim, PetscScalar* conservedValues, PetscReal gamma, PetscReal Rgas){
    PetscReal density = conservedValues[RHO];
    PetscReal totalEnergy = conservedValues[RHOE]/density;

    // Get the velocity in this direction
    PetscReal speedSquare = 0.0;
    for (PetscInt d =0; d < dim; d++){
        speedSquare += PetscSqr(conservedValues[RHOU + d]/density);
    }

    // assumed eos
    PetscReal internalEnergy = (totalEnergy) - 0.5 * speedSquare;
    PetscReal p = (gamma - 1.0)*density*internalEnergy;

    PetscReal T = p/(Rgas*density);
    return T;
}

static PetscErrorCode DiffusionSource(DM dm, PetscReal time, Vec locXVec, Vec globFVec, void *ctx){
    // Call the flux calculation
    PetscErrorCode ierr;

    ProblemSetup *setup = (ProblemSetup *)ctx;


     ierr = DMPlexTSComputeRHSFunctionFVM(dm, time, locXVec, globFVec, setup->flowData);CHKERRQ(ierr);

    Constants constants = setup->constants;

    PetscInt dim;
    ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);

    // Get the locXArray
    const PetscScalar *locXArray;
    ierr = VecGetArrayRead(locXVec, &locXArray);CHKERRQ(ierr);

    // Get the fvm face and cell geometry
    Vec cellGeomVec = NULL;/* vector of structs related to cell geometry*/
    Vec faceGeomVec = NULL;/* vector of structs related to face geometry*/
    ierr = DMPlexGetGeometryFVM(dm, &faceGeomVec, &cellGeomVec, NULL);CHKERRQ(ierr);

    // get the dm for each geom type
    DM dmFaceGeom, dmCellGeom;
    ierr = VecGetDM(faceGeomVec, &dmFaceGeom);CHKERRQ(ierr);
    ierr = VecGetDM(cellGeomVec, &dmCellGeom);CHKERRQ(ierr);

    // extract the arrays for the face and cell geom, along with their dm
    const PetscScalar *faceGeomArray, *cellGeomArray;
    ierr = VecGetArrayRead(cellGeomVec, &cellGeomArray);CHKERRQ(ierr);
    ierr = VecGetArrayRead(faceGeomVec, &faceGeomArray);CHKERRQ(ierr);

    // Obtaining local cell and face ownership
    PetscInt faceStart, faceEnd;
    PetscInt cellStart, cellEnd;
    ierr = DMPlexGetHeightStratum(dm, 1, &faceStart, &faceEnd);CHKERRQ(ierr);
    ierr = DMPlexGetHeightStratum(dm, 0, &cellStart, &cellEnd);CHKERRQ(ierr);

    // get the fvm and the number of fields
    PetscFV fvm;
    ierr = DMGetField(dm,0, NULL, (PetscObject*)&fvm);CHKERRQ(ierr);
    PetscInt components;
    ierr = PetscFVGetNumComponents(fvm, &components);CHKERRQ(ierr);

    // get the ghost label
    DMLabel ghostLabel;
    ierr = DMGetLabel(dm, "ghost", &ghostLabel);CHKERRQ(ierr);

    // extract the localFArray from the locFVec
    PetscScalar *fa;
    Vec locFVec;
    ierr = DMGetLocalVector(dm, &locFVec);CHKERRQ(ierr);
    ierr = VecZeroEntries(locFVec);CHKERRQ(ierr);
    ierr = VecGetArray(locFVec, &fa);CHKERRQ(ierr);

    // march over each face
    for (PetscInt face = faceStart; face < faceEnd; ++face) {
        PetscFVFaceGeom       *fg;
        PetscFVCellGeom       *cgL, *cgR;

        // make sure that this is a valid face to check
        PetscInt  ghost, nsupp, nchild;
        ierr = DMLabelGetValue(ghostLabel, face, &ghost);CHKERRQ(ierr);
        ierr = DMPlexGetSupportSize(dm, face, &nsupp);CHKERRQ(ierr);
        ierr = DMPlexGetTreeChildren(dm, face, &nchild, NULL);CHKERRQ(ierr);
        if (ghost >= 0 || nsupp > 2 || nchild > 0){
            continue;// skip this face
        }

        // get the face geometry
        ierr = DMPlexPointLocalRead(dmFaceGeom, face, faceGeomArray, &fg);CHKERRQ(ierr);

        // Get the left and right cells for this face
        const PetscInt        *faceCells;
        ierr = DMPlexGetSupport(dm, face, &faceCells);CHKERRQ(ierr);

        // get the cell geom for the left and right faces
        ierr = DMPlexPointLocalRead(dmCellGeom, faceCells[0], cellGeomArray, &cgL);CHKERRQ(ierr);
        ierr = DMPlexPointLocalRead(dmCellGeom, faceCells[1], cellGeomArray, &cgR);CHKERRQ(ierr);

        PetscInt f = 0;

        // extract the field values
        PetscScalar *xL, *xR,
        ierr = DMPlexPointLocalFieldRead(dm, faceCells[0], f, locXArray, &xL);CHKERRQ(ierr);
        ierr = DMPlexPointLocalFieldRead(dm, faceCells[1], f, locXArray, &xR);CHKERRQ(ierr);

        // compute the temperature at the left and right nodes
        PetscReal TL = computeTemperature(dim, xL, constants.gamma, constants.Rgas);
        PetscReal TR = computeTemperature(dim, xR, constants.gamma, constants.Rgas);

        // Compute the grad
        PetscReal fluxArea = 0.0;
        for (PetscInt d = 0; d < dim; ++d){
            // compute dx in that direction
            PetscReal dx  = cgR->centroid[d] - cgL->centroid[d];
            fluxArea += -constants.k *fg->normal[d]* (TR - TL) * dx /(dx*dx + 1.0E-60);
        }

        // Add to the source terms of f
        PetscScalar    *fL = NULL, *fR = NULL;
        ierr = DMLabelGetValue(ghostLabel,faceCells[0],&ghost);CHKERRQ(ierr);
        if (ghost <= 0) {ierr = DMPlexPointLocalFieldRef(dm, faceCells[0], f, fa, &fL);CHKERRQ(ierr);}
        ierr = DMLabelGetValue(ghostLabel,faceCells[1],&ghost);CHKERRQ(ierr);
        if (ghost <= 0) {ierr = DMPlexPointLocalFieldRef(dm, faceCells[1], f, fa, &fR);CHKERRQ(ierr);}

        if(fL){
            fL[RHOE] -= fluxArea/cgL->volume;
        }
        if(fR){
            fR[RHOE] += fluxArea/cgR->volume;
        }
    }

    // Add the new locFVec to the globFVec
    ierr = VecRestoreArray(locFVec, &fa);CHKERRQ(ierr);
    ierr = DMLocalToGlobalBegin(dm, locFVec, INSERT_VALUES, globFVec);CHKERRQ(ierr);
    ierr = DMLocalToGlobalEnd(dm, locFVec, INSERT_VALUES, globFVec);CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(dm, &locFVec);CHKERRQ(ierr);

    // restore the arrays
    ierr = VecRestoreArrayRead(cellGeomVec, &cellGeomArray);CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(faceGeomVec, &faceGeomArray);CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(locXVec, &locXArray);CHKERRQ(ierr);

    PetscFunctionReturn(0);
}

static PetscErrorCode PhysicsBoundary_Euler(PetscReal time, const PetscReal *c, const PetscReal *n, const PetscScalar *a_xI, PetscScalar *a_xG, void *ctx) {
    PetscFunctionBeginUser;
    Constants *constants = (Constants *)ctx;


    InitialConditions(constants->dim, time, c, 0, a_xG, ctx);
    PetscFunctionReturn(0);
}

static PetscErrorCode PhysicsBoundary_Mirror(PetscReal time, const PetscReal *c, const PetscReal *n, const PetscScalar *a_xI, PetscScalar *a_xG, void *ctx) {
    PetscFunctionBeginUser;
    Constants *constants = (Constants *)ctx;

    // Offset the calc assuming the cells are square
    for(PetscInt f =0; f < RHOU + constants->dim; f++){
        a_xG[f] = a_xI[f];
    }
    PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
    // Setup the problem
    Constants constants;

    // sub sonic
    constants.dim = 2;
    constants.L = 1.0;
    constants.Rgas = 287.0;
    constants.gamma = 1.4;
    constants.k = 0.02514;
    constants.Rgas = 287.0;

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

    // Create a mesh
    // hard code the problem setup
    PetscReal start[] = {0.0, 0.0};
    PetscReal end[] = {constants.L, constants.L};
    PetscReal nxHeight = 10;
    PetscInt nx[] = {nxHeight, nxHeight};
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
    const PetscInt idsLeft[]= {2, 4};
    ierr = PetscDSAddBoundary(prob, DM_BC_NATURAL_RIEMANN, "wall left", "Face Sets", 0, 0, NULL, (void (*)(void))PhysicsBoundary_Euler, NULL, 2, idsLeft, &constants);CHKERRQ(ierr);

    const PetscInt idsTop[]= {1, 3};
    ierr = PetscDSAddBoundary(prob, DM_BC_NATURAL_RIEMANN, "top/bottom", "Face Sets", 0, 0, NULL, (void (*)(void))PhysicsBoundary_Mirror, NULL, 2, idsTop, &constants);CHKERRQ(ierr);


    // Complete the problem setup
    ierr = CompressibleFlow_CompleteProblemSetup(flowData, ts);
    CHKERRABORT(PETSC_COMM_WORLD, ierr);

    // Override the flow calc for now
    ierr = DMTSSetRHSFunctionLocal(flowData->dm, DiffusionSource, &problemSetup);CHKERRQ(ierr);

    // Name the flow field
    ierr = PetscObjectSetName(((PetscObject)flowData->flowField), "Numerical Solution");
    CHKERRABORT(PETSC_COMM_WORLD, ierr);

    // Setup the TS
    ierr = TSSetFromOptions(ts);
    CHKERRABORT(PETSC_COMM_WORLD, ierr);
    ierr = TSMonitorSet(ts, MonitorError, &constants, NULL);CHKERRQ(ierr);
    CHKERRABORT(PETSC_COMM_WORLD, ierr);

    // set the initial conditions
    PetscErrorCode     (*func[2]) (PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar *u, void *ctx) = {InitialConditions};
    void* ctxs[1] ={&constants};
    ierr    = DMProjectFunction(flowData->dm,0.0,func,ctxs,INSERT_ALL_VALUES,flowData->flowField);CHKERRQ(ierr);

    // for the mms, add the exact solution
    ierr = PetscDSSetExactSolution(prob, 0, InitialConditions, &constants);CHKERRQ(ierr);

    // Output the mesh
    ierr = DMViewFromOptions(flowData->dm, NULL, "-dm_view");CHKERRQ(ierr);

    // Compute the end time so it goes around once
    // compute dt
    PetscReal dxMin = .001;
    PetscReal cp = constants.gamma*constants.Rgas/(constants.gamma - 1);

    PetscReal alpha = PetscAbs(constants.k/ (1.0 * cp));
    double dt_conduc = PetscSqr(dxMin) / alpha;

    TSSetTimeStep(ts, dt_conduc);
    TSSetMaxTime(ts, 1.0);
    TSSetMaxSteps(ts, 100);

    PetscDSView(prob, PETSC_VIEWER_STDOUT_WORLD);

    ierr = TSSolve(ts,flowData->flowField);CHKERRQ(ierr);

    FlowDestroy(&flowData);
    TSDestroy(&ts);

    return PetscFinalize();

}