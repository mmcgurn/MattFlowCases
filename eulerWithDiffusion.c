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

static PetscReal ComputeTExact( PetscReal time, const PetscReal xyz[], Constants *constants, PetscReal rho){

    PetscReal cp = constants->gamma*constants->Rgas/(constants->gamma - 1);

    PetscReal alpha = constants->k/(rho*cp);

    PetscReal Tinitial = 100.0;
    time += 0.0001/2.0;
    PetscReal T = 0.0;
    for(PetscReal n =1; n < 20000; n ++){
        PetscReal Bn = -Tinitial*2.0*(-1.0 + PetscPowReal(-1.0, n))/(n*PETSC_PI);
        T += Bn*PetscSinReal(n * PETSC_PI*xyz[0]/constants->L)*PetscExpReal(-n*n*PETSC_PI*PETSC_PI*alpha*time/(PetscSqr(constants->L)));
    }

    return xyz[0];
}

static PetscErrorCode InitialConditions(PetscInt dim, PetscReal time, const PetscReal xyz[], PetscInt Nf, PetscScalar *node, void *ctx) {
    PetscFunctionBeginUser;

    Constants *constants = (Constants *)ctx;

    PetscReal T = 300.0 +   ComputeTExact(time, xyz, constants, 1.0);

    PetscReal u = 0.0;
    PetscReal v = 0.0;
    PetscReal rho = 1.0;
    PetscReal p= rho*constants->Rgas*T;
    PetscReal e = p/((constants->gamma - 1.0)*rho);
    PetscReal eT = e + 0.5*(u*u + v*v);

    node[RHO] = rho;
    node[RHOE] = rho*eT;
    node[RHOU + 0] = rho*u;
    node[RHOU + 1] = rho*v;

    PetscFunctionReturn(0);
}


PetscErrorCode DMPlexComputeGradients(DM dm, DM dmGrad, Vec faceGeometryVec, Vec cellGeometryVec, Vec locXVec, Vec locGradVec, PetscInt dof)
{
    PetscErrorCode     ierr;

    PetscFunctionBeginUser;
    PetscInt           dim;
    ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
    DMLabel            ghostLabel;
    ierr = DMGetLabel(dm, "ghost", &ghostLabel);CHKERRQ(ierr);

    // get the dm for the face, cell geometry, and grad
    DM                 dmFace, dmCell;
    ierr = VecGetDM(faceGeometryVec, &dmFace);CHKERRQ(ierr);
    ierr = VecGetDM(cellGeometryVec, &dmCell);CHKERRQ(ierr);
    ierr = VecGetDM(locGradVec, &dmGrad);CHKERRQ(ierr);

    // open the face, geom, and x array
    const PetscScalar *faceGeometryArray, *cellGeometryArray, *xArray;
    ierr = VecGetArrayRead(faceGeometryVec, &faceGeometryArray);CHKERRQ(ierr);
    ierr = VecGetArrayRead(cellGeometryVec, &cellGeometryArray);CHKERRQ(ierr);
    ierr = VecGetArrayRead(locXVec, &xArray);CHKERRQ(ierr);

    // initialize the grad array
    PetscScalar       *gradArray;
    ierr = VecZeroEntries(locGradVec);CHKERRQ(ierr);
    ierr = VecGetArray(locGradVec, &gradArray);CHKERRQ(ierr);

    // Obtaining local cell and face ownership
    PetscInt faceStart, faceEnd;
    ierr = DMPlexGetHeightStratum(dm, 1, &faceStart, &faceEnd);CHKERRQ(ierr);

    /* Reconstruct gradients */
    for (PetscInt face = faceStart; face < faceEnd; ++face) {
        const PetscInt        *cells;
        PetscFVFaceGeom       *fg;
        PetscScalar           *cx[2];
        PetscScalar           *cgrad[2];
        PetscBool              boundary;
        PetscInt               ghost, c, pd, d, numChildren, numCells;

        ierr = DMLabelGetValue(ghostLabel, face, &ghost);CHKERRQ(ierr);
        ierr = DMIsBoundaryPoint(dm, face, &boundary);CHKERRQ(ierr);
        ierr = DMPlexGetTreeChildren(dm, face, &numChildren, NULL);CHKERRQ(ierr);
        if (ghost >= 0 || boundary || numChildren){
            continue;
        }
        ierr = DMPlexGetSupportSize(dm, face, &numCells);CHKERRQ(ierr);
        if (numCells != 2){
            SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_PLIB, "facet %d has %d support points: expected 2",face,numCells);
        }
        ierr = DMPlexGetSupport(dm, face, &cells);CHKERRQ(ierr);
        ierr = DMPlexPointLocalRead(dmFace, face, faceGeometryArray, &fg);CHKERRQ(ierr);

        // extract the current value and the location to put the gradient
        for (c = 0; c < 2; ++c) {
            ierr = DMPlexPointLocalRead(dm, cells[c], xArray, &cx[c]);CHKERRQ(ierr);
            ierr = DMPlexPointLocalRef(dmGrad, cells[c], gradArray, &cgrad[c]);CHKERRQ(ierr);
        }
        for (pd = 0; pd < dof; ++pd) {
            PetscScalar delta = cx[1][pd] - cx[0][pd];

            for (d = 0; d < dim; ++d) {
                if (cgrad[0]) cgrad[0][pd*dim+d] += fg->grad[0][d] * delta;
                if (cgrad[1]) cgrad[1][pd*dim+d] -= fg->grad[1][d] * delta;
            }
        }
    }

    ierr = VecRestoreArrayRead(faceGeometryVec, &faceGeometryArray);CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(locXVec, &xArray);CHKERRQ(ierr);
    ierr = VecRestoreArray(locGradVec, &gradArray);CHKERRQ(ierr);
    PetscFunctionReturn(0);
}



static PetscErrorCode MonitorError(TS ts, PetscInt step, PetscReal time, Vec u, void *ctx) {
    PetscFunctionBeginUser;
    PetscErrorCode     ierr;

    FlowData flowData = (FlowData)ctx;

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
    ierr = VecViewFromOptions(exactVec, NULL, "-sol_view");
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
    ierr = VecViewFromOptions(exactVec, NULL, "-sol_view");
    CHKERRQ(ierr);

    if (flowData->auxDm) {
        ierr = DMSetOutputSequenceNumber(flowData->auxDm, step, time);CHKERRQ(ierr);
        ierr = VecViewFromOptions(flowData->auxField, NULL, "-sol_view");CHKERRQ(ierr);
        VecView(flowData->auxField, PETSC_VIEWER_STDOUT_WORLD);
    }

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
//


static PetscErrorCode DiffusionSource(DM dm, PetscReal time, Vec locXVec, Vec globFVec, void *ctx){
    // Call the flux calculation
    PetscErrorCode ierr;

    ProblemSetup *setup = (ProblemSetup *)ctx;

    ierr = DMPlexTSComputeRHSFunctionFVM(dm, time, locXVec, globFVec, setup->flowData);CHKERRQ(ierr);

    Constants constants = setup->constants;


    // convert the dm assuming that it is a plex
    DM plex;
    ierr = DMConvert(dm, DMPLEX, &plex);CHKERRQ(ierr);

    // get the fvm field assuming that it is the first
    PetscFV fvm;
    ierr = DMGetField(dm,0, NULL, (PetscObject*)&fvm);CHKERRQ(ierr);

    PetscInt dim;
    ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);

    // Get the locXArray
    const PetscScalar *locXArray;
    ierr = VecGetArrayRead(locXVec, &locXArray);CHKERRQ(ierr);

    // Get the fvm face and cell geometry
    Vec cellGeomVec = NULL;/* vector of structs related to cell geometry*/
    Vec faceGeomVec = NULL;/* vector of structs related to face geometry*/
    DM gradDM = NULL; /* dm holding the grad information */

    // extract the fvm data
    ierr = DMPlexGetDataFVM(plex, fvm, &cellGeomVec, &faceGeomVec, &gradDM);CHKERRQ(ierr);

    // Size T vector



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

    // create a global and local grad vector
    Vec gradGlobalVec, gradLocalVec;
    ierr = DMCreateGlobalVector(gradDM, &gradGlobalVec);CHKERRQ(ierr);

    // compute the global grad values
    ierr = DMPlexReconstructGradientsFVM(plex, locXVec, gradGlobalVec);CHKERRQ(ierr);

    // Map to a local grad vector
    ierr = DMCreateLocalVector(gradDM, &gradLocalVec);CHKERRQ(ierr);
    ierr = DMGlobalToLocalBegin(gradDM, gradGlobalVec, INSERT_VALUES, gradLocalVec);CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(gradDM, gradGlobalVec, INSERT_VALUES, gradLocalVec);CHKERRQ(ierr);

    // access the local vector
    const PetscScalar *localGradArray;
    ierr = VecGetArrayRead(gradLocalVec,&localGradArray);CHKERRQ(ierr);

    // march over each face
    for (PetscInt face = faceStart; face < faceEnd; ++face) {
        PetscFVFaceGeom       *fg;
        PetscFVCellGeom       *cgL, *cgR;
        const PetscScalar           *gradL;
        const PetscScalar           *gradR;

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

        // extract the cell grad
        ierr = DMPlexPointLocalRead(gradDM, faceCells[0], localGradArray, &gradL);CHKERRQ(ierr);
        ierr = DMPlexPointLocalRead(gradDM, faceCells[1], localGradArray, &gradR);CHKERRQ(ierr);

        PetscInt f = 0;

        // extract the field values
        PetscScalar *xL, *xR,
        ierr = DMPlexPointLocalFieldRead(dm, faceCells[0], f, locXArray, &xL);CHKERRQ(ierr);
        ierr = DMPlexPointLocalFieldRead(dm, faceCells[1], f, locXArray, &xR);CHKERRQ(ierr);

        // compute the temperature at the left and right nodes
        PetscReal TL = computeTemperature(dim, xL, constants.gamma, constants.Rgas);
        PetscReal TR = computeTemperature(dim, xR, constants.gamma, constants.Rgas);

        if(fg->centroid[0] < (0.0035 + 1E-8) && fg->centroid[0] > (0.0035 - 1E-8) ){
            if(fg->centroid[1] < (0.003) && fg->centroid[1] >  0.0025 ){
                puts("debug");
            }
        }

        // Compute the grad
        PetscReal fluxArea = 0.0;
        PetscInt dof = 1;
        for (PetscInt d = 0; d < dim; ++d){
            // compute dx in that direction
            PetscReal dx  = cgR->centroid[d] - cgL->centroid[d];
            PetscReal grad  = 0.5*(gradL[dof*dim + d] + gradR[dof*dim + d]);
            PetscReal gradFactor = (TR - TL)/dx;
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
    ierr = VecRestoreArrayRead(gradLocalVec, &localGradArray);CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(cellGeomVec, &cellGeomArray);CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(faceGeomVec, &faceGeomArray);CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(locXVec, &locXArray);CHKERRQ(ierr);

    // destroy grad vectors
    ierr = VecDestroy(&gradGlobalVec);CHKERRQ(ierr);
    ierr = VecDestroy(&gradLocalVec);CHKERRQ(ierr);

    PetscFunctionReturn(0);
}

static PetscErrorCode PhysicsBoundary_Euler(PetscReal time, const PetscReal *c, const PetscReal *n, const PetscScalar *a_xI, PetscScalar *a_xG, void *ctx) {
    PetscFunctionBeginUser;
    Constants *constants = (Constants *)ctx;

    // compute the centroid location of the real cell
    // Offset the calc assuming the cells are square
    PetscReal x[3];
    for(PetscInt i =0; i < constants->dim; i++){
        x[i] = c[i] - n[i]*0.5;
    }

    // compute the temperature
    PetscReal Tinside = 300.0 +   ComputeTExact(time, x, constants, 1.0);
    PetscReal boundaryValue = 300.0;

    PetscReal T = boundaryValue - (Tinside - boundaryValue);

    PetscReal u = 0.0;
    PetscReal v = 0.0;
    PetscReal rho = 1.0;
    PetscReal p= rho*constants->Rgas*T;
    PetscReal e = p/((constants->gamma - 1.0)*rho);
    PetscReal eT = e + 0.5*(u*u + v*v);

    a_xG[RHO] = rho;
    a_xG[RHOE] = rho*eT;
    a_xG[RHOU + 0] = rho*u;
    a_xG[RHOU + 1] = rho*v;

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
    constants.L = 0.01;
    constants.gamma = 1.4;
    constants.k = 0.3;
    constants.Rgas = 1;//287.0;

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

    ierr = FlowRegisterAuxField(flowData, "Temperature", "T", 1);CHKERRQ(ierr);

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
    ierr = TSMonitorSet(ts, MonitorError, problemSetup.flowData, NULL);CHKERRQ(ierr);
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
    PetscReal dxMin = constants.L/(10 * PetscPowRealInt(2, 2));

    PetscReal alpha = PetscAbs(constants.k/ (1.0 * 1.0));
    double dt_conduc = 1.0*PetscSqr(dxMin) / alpha;

    PetscPrintf(PETSC_COMM_WORLD, "dt_conduc: %f\n", dt_conduc);
    TSSetTimeStep(ts, dt_conduc);
    TSSetMaxTime(ts, 0.0001);
    TSSetMaxSteps(ts, 1);

    PetscDSView(prob, PETSC_VIEWER_STDOUT_WORLD);

    ierr = TSSolve(ts,flowData->flowField);CHKERRQ(ierr);

    FlowDestroy(&flowData);
    TSDestroy(&ts);

    return PetscFinalize();

}