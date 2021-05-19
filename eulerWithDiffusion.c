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

    PetscReal cv = constants->gamma*constants->Rgas/(constants->gamma - 1) - constants->Rgas;

    PetscReal alpha = constants->k/(rho*cv);
    PetscReal Tinitial = 100.0;
    PetscReal T = 300.0;
    for(PetscReal n =1; n < 2000; n ++){
        PetscReal Bn = -Tinitial*2.0*(-1.0 + PetscPowReal(-1.0, n))/(n*PETSC_PI);
        T += Bn*PetscSinReal(n * PETSC_PI*xyz[0]/constants->L)*PetscExpReal(-n*n*PETSC_PI*PETSC_PI*alpha*time/(PetscSqr(constants->L)));
    }

//    PetscReal TOther = 0.0;
//    for(PetscReal n =1; n < 2000000; n ++){
//        TOther += 4.0*Tinitial/((2.0*n -1.0) * PETSC_PI) * PetscSinReal((2.0*n - 1) * PETSC_PI*xyz[0]/constants->L) * PetscExpReal(-alpha* PetscSqr(2.0*n-1)*PETSC_PI*PETSC_PI*time/PetscSqr(constants->L));
//    }
//
//    PetscReal TOther2 = 0.0;
//    for(PetscReal n =1; n < 2000000; n += 2){// odd only
//        TOther2 += 4.0*Tinitial/(n*PETSC_PI)* PetscExpReal(-n*n*PETSC_PI*PETSC_PI*alpha*time/PetscSqr(constants->L))*sin(n*PETSC_PI*xyz[0]/constants->L);
//    }


    return T;
}

static PetscErrorCode InitialConditions(PetscInt dim, PetscReal time, const PetscReal xyz[], PetscInt Nf, PetscScalar *node, void *ctx) {
    PetscFunctionBeginUser;

    Constants *constants = (Constants *)ctx;

    PetscReal T = ComputeTExact(time, xyz, constants, 1.0);

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

//
//PetscErrorCode DMPlexComputeGradients(DM dm, DM dmGrad, Vec faceGeometryVec, Vec cellGeometryVec, Vec locXVec, Vec locGradVec, PetscInt dof)
//{
//    PetscErrorCode     ierr;
//
//    PetscFunctionBeginUser;
//    PetscInt           dim;
//    ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
//    DMLabel            ghostLabel;
//    ierr = DMGetLabel(dm, "ghost", &ghostLabel);CHKERRQ(ierr);
//
//    // get the dm for the face, cell geometry, and grad
//    DM                 dmFace, dmCell;
//    ierr = VecGetDM(faceGeometryVec, &dmFace);CHKERRQ(ierr);
//    ierr = VecGetDM(cellGeometryVec, &dmCell);CHKERRQ(ierr);
//    ierr = VecGetDM(locGradVec, &dmGrad);CHKERRQ(ierr);
//
//    // open the face, geom, and x array
//    const PetscScalar *faceGeometryArray, *cellGeometryArray, *xArray;
//    ierr = VecGetArrayRead(faceGeometryVec, &faceGeometryArray);CHKERRQ(ierr);
//    ierr = VecGetArrayRead(cellGeometryVec, &cellGeometryArray);CHKERRQ(ierr);
//    ierr = VecGetArrayRead(locXVec, &xArray);CHKERRQ(ierr);
//
//    // initialize the grad array
//    PetscScalar       *gradArray;
//    ierr = VecZeroEntries(locGradVec);CHKERRQ(ierr);
//    ierr = VecGetArray(locGradVec, &gradArray);CHKERRQ(ierr);
//
//    // Obtaining local cell and face ownership
//    PetscInt faceStart, faceEnd;
//    ierr = DMPlexGetHeightStratum(dm, 1, &faceStart, &faceEnd);CHKERRQ(ierr);
//
//    /* Reconstruct gradients */
//    for (PetscInt face = faceStart; face < faceEnd; ++face) {
//        const PetscInt        *cells;
//        PetscFVFaceGeom       *fg;
//        PetscScalar           *cx[2];
//        PetscScalar           *cgrad[2];
//        PetscBool              boundary;
//        PetscInt               ghost, c, pd, d, numChildren, numCells;
//
//        ierr = DMLabelGetValue(ghostLabel, face, &ghost);CHKERRQ(ierr);
//        ierr = DMIsBoundaryPoint(dm, face, &boundary);CHKERRQ(ierr);
//        ierr = DMPlexGetTreeChildren(dm, face, &numChildren, NULL);CHKERRQ(ierr);
//        if (ghost >= 0 || boundary || numChildren){
//            continue;
//        }
//        ierr = DMPlexGetSupportSize(dm, face, &numCells);CHKERRQ(ierr);
//        if (numCells != 2){
//            SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_PLIB, "facet %d has %d support points: expected 2",face,numCells);
//        }
//        ierr = DMPlexGetSupport(dm, face, &cells);CHKERRQ(ierr);
//        ierr = DMPlexPointLocalRead(dmFace, face, faceGeometryArray, &fg);CHKERRQ(ierr);
//
//        // extract the current value and the location to put the gradient
//        for (c = 0; c < 2; ++c) {
//            ierr = DMPlexPointLocalRead(dm, cells[c], xArray, &cx[c]);CHKERRQ(ierr);
//            ierr = DMPlexPointLocalRef(dmGrad, cells[c], gradArray, &cgrad[c]);CHKERRQ(ierr);
//        }
//        for (pd = 0; pd < dof; ++pd) {
//            PetscScalar delta = cx[1][pd] - cx[0][pd];
//
//            for (d = 0; d < dim; ++d) {
//                if (cgrad[0]) cgrad[0][pd*dim+d] += fg->grad[0][d] * delta;
//                if (cgrad[1]) cgrad[1][pd*dim+d] -= fg->grad[1][d] * delta;
//            }
//        }
//    }
//
//    ierr = VecRestoreArrayRead(faceGeometryVec, &faceGeometryArray);CHKERRQ(ierr);
//    ierr = VecRestoreArrayRead(locXVec, &xArray);CHKERRQ(ierr);
//    ierr = VecRestoreArray(locGradVec, &gradArray);CHKERRQ(ierr);
//    PetscFunctionReturn(0);
//}

static PetscReal computeTemperature(PetscInt dim, const PetscScalar* conservedValues, PetscReal gamma, PetscReal Rgas){
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

static PetscErrorCode UpdateAuxFields(TS ts, Vec locXVec, void*context){
    PetscFunctionBeginUser;

    PetscErrorCode ierr;
    FlowData flowData = (FlowData)context;
    EulerFlowData * flowParameters = (EulerFlowData *)flowData->data;

    // Extract the cell geometry, and the dm that holds the information
    Vec cellgeom;
    DM dmCell;
    const PetscScalar *cgeom;
    ierr = DMPlexGetGeometryFVM(flowData->dm, NULL, &cellgeom, NULL);CHKERRQ(ierr);
    ierr = VecGetDM(cellgeom, &dmCell);CHKERRQ(ierr);
    ierr = VecGetArrayRead(cellgeom, &cgeom);CHKERRQ(ierr);

    // Get the cell start and end for the fv cells
    PetscInt cellStart, cellEnd;
    ierr = DMPlexGetHeightStratum(flowData->dm, 0, &cellStart, &cellEnd);CHKERRQ(ierr);

    const PetscScalar      *locFlowFieldArray;
    ierr = VecGetArrayRead(locXVec, &locFlowFieldArray);CHKERRQ(ierr);

    PetscScalar     *localAuxFlowFieldArray;
    ierr = VecGetArray(flowData->auxField, &localAuxFlowFieldArray);CHKERRQ(ierr);

    // Get the cell dim
    PetscInt dim;
    ierr = DMGetDimension(flowData->dm, &dim);CHKERRQ(ierr);

    // March over each cell volume
    for (PetscInt c = cellStart; c < cellEnd; ++c) {
        PetscFVCellGeom       *cg;
        const PetscReal           *fc;
        PetscReal           *afc;

        ierr = DMPlexPointLocalRead(dmCell, c, cgeom, &cg);CHKERRQ(ierr);
        ierr = DMPlexPointLocalFieldRead(flowData->dm, c, 0, locFlowFieldArray, &fc);CHKERRQ(ierr);
        ierr = DMPlexPointLocalFieldRef(flowData->auxDm, c, 0, localAuxFlowFieldArray, &afc);CHKERRQ(ierr);

        if(afc) {
            // Compute the temperature
            afc[0] = computeTemperature(dim, fc, flowParameters->gamma, flowParameters->Rgas);
        }
    }

    // restore the cell geometry
    ierr = VecRestoreArrayRead(cellgeom, &cgeom);CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(locXVec, &locFlowFieldArray);CHKERRQ(ierr);
    ierr = VecRestoreArray(flowData->auxField, &localAuxFlowFieldArray);CHKERRQ(ierr);

//    VecView(flowData->auxField, PETSC_VIEWER_STDOUT_WORLD);


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

    ierr = FlowViewFromOptions(flowData, "-sol_view");
    CHKERRQ(ierr);

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

    ierr = VecDestroy(&exactVec);
    CHKERRQ(ierr);
    PetscFunctionReturn(0);
}

//
//static PetscErrorCode DiffusionSource(DM dm, PetscReal time, Vec locXVec, Vec globFVec, void *ctx){
//    // Call the flux calculation
//    PetscErrorCode ierr;
//
//    ProblemSetup *setup = (ProblemSetup *)ctx;
//
//
//    ierr = DMPlexTSComputeRHSFunctionFVM(dm, time, locXVec, globFVec, setup->flowData);CHKERRQ(ierr);
//
//    Constants constants = setup->constants;
//
//    PetscInt dim;
//    ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
//
//    // Get the locXArray
//    const PetscScalar *locXArray;
//    ierr = VecGetArrayRead(locXVec, &locXArray);CHKERRQ(ierr);
//
//    // Get the fvm face and cell geometry
//    Vec cellGeomVec = NULL;/* vector of structs related to cell geometry*/
//    Vec faceGeomVec = NULL;/* vector of structs related to face geometry*/
//    ierr = DMPlexGetGeometryFVM(dm, &faceGeomVec, &cellGeomVec, NULL);CHKERRQ(ierr);
//
//    // get the dm for each geom type
//    DM dmFaceGeom, dmCellGeom;
//    ierr = VecGetDM(faceGeomVec, &dmFaceGeom);CHKERRQ(ierr);
//    ierr = VecGetDM(cellGeomVec, &dmCellGeom);CHKERRQ(ierr);
//
//    // extract the arrays for the face and cell geom, along with their dm
//    const PetscScalar *faceGeomArray, *cellGeomArray;
//    ierr = VecGetArrayRead(cellGeomVec, &cellGeomArray);CHKERRQ(ierr);
//    ierr = VecGetArrayRead(faceGeomVec, &faceGeomArray);CHKERRQ(ierr);
//
//    // Obtaining local cell and face ownership
//    PetscInt faceStart, faceEnd;
//    PetscInt cellStart, cellEnd;
//    ierr = DMPlexGetHeightStratum(dm, 1, &faceStart, &faceEnd);CHKERRQ(ierr);
//    ierr = DMPlexGetHeightStratum(dm, 0, &cellStart, &cellEnd);CHKERRQ(ierr);
//
//    // get the fvm and the number of fields
//    PetscFV fvm;
//    ierr = DMGetField(dm,0, NULL, (PetscObject*)&fvm);CHKERRQ(ierr);
//    PetscInt components;
//    ierr = PetscFVGetNumComponents(fvm, &components);CHKERRQ(ierr);
//
//    // get the ghost label
//    DMLabel ghostLabel;
//    ierr = DMGetLabel(dm, "ghost", &ghostLabel);CHKERRQ(ierr);
//
//    // extract the localFArray from the locFVec
//    PetscScalar *fa;
//    Vec locFVec;
//    ierr = DMGetLocalVector(dm, &locFVec);CHKERRQ(ierr);
//    ierr = VecZeroEntries(locFVec);CHKERRQ(ierr);
//    ierr = VecGetArray(locFVec, &fa);CHKERRQ(ierr);
//
//    // march over each face
//    for (PetscInt face = faceStart; face < faceEnd; ++face) {
//        PetscFVFaceGeom       *fg;
//        PetscFVCellGeom       *cgL, *cgR;
//
//        // make sure that this is a valid face to check
//        PetscInt  ghost, nsupp, nchild;
//        ierr = DMLabelGetValue(ghostLabel, face, &ghost);CHKERRQ(ierr);
//        ierr = DMPlexGetSupportSize(dm, face, &nsupp);CHKERRQ(ierr);
//        ierr = DMPlexGetTreeChildren(dm, face, &nchild, NULL);CHKERRQ(ierr);
//        if (ghost >= 0 || nsupp > 2 || nchild > 0){
//            continue;// skip this face
//        }
//
//        // get the face geometry
//        ierr = DMPlexPointLocalRead(dmFaceGeom, face, faceGeomArray, &fg);CHKERRQ(ierr);
//
//        // Get the left and right cells for this face
//        const PetscInt        *faceCells;
//        ierr = DMPlexGetSupport(dm, face, &faceCells);CHKERRQ(ierr);
//
//        // get the cell geom for the left and right faces
//        ierr = DMPlexPointLocalRead(dmCellGeom, faceCells[0], cellGeomArray, &cgL);CHKERRQ(ierr);
//        ierr = DMPlexPointLocalRead(dmCellGeom, faceCells[1], cellGeomArray, &cgR);CHKERRQ(ierr);
//
//        PetscInt f = 0;
//
//        // extract the field values
//        PetscScalar *xL, *xR,
//        ierr = DMPlexPointLocalFieldRead(dm, faceCells[0], f, locXArray, &xL);CHKERRQ(ierr);
//        ierr = DMPlexPointLocalFieldRead(dm, faceCells[1], f, locXArray, &xR);CHKERRQ(ierr);
//
//        // compute the temperature at the left and right nodes
//        PetscReal TL = computeTemperature(dim, xL, constants.gamma, constants.Rgas);
//        PetscReal TR = computeTemperature(dim, xR, constants.gamma, constants.Rgas);
//
//        // Compute the ds vector
//        PetscReal dsVec[3];
//        PetscReal ds = 0.0;
//        PetscReal dsDotNorm = 0.0;
//        PetscReal normalArea = 0.0;
//        for (PetscInt d = 0; d < dim; ++d) {
//            dsVec[d] = cgR->centroid[d] - cgL->centroid[d];
//            ds += PetscSqr(dsVec[d]);
//            dsDotNorm += dsVec[d]*fg->normal[d];
//            normalArea += PetscSqr(fg->normal[d]);
//        }
//        ds = PetscSqrtReal(ds);
//        normalArea = PetscSqrtReal(normalArea);
//
//        // Compute the normal flux
//        PetscReal normalFlux =  -constants.k*normalArea * (TR - TL)/ds;
//
//        // Add to the source terms of f
//        PetscScalar    *fL = NULL, *fR = NULL;
//        ierr = DMLabelGetValue(ghostLabel,faceCells[0],&ghost);CHKERRQ(ierr);
//        if (ghost <= 0) {ierr = DMPlexPointLocalFieldRef(dm, faceCells[0], f, fa, &fL);CHKERRQ(ierr);}
//        ierr = DMLabelGetValue(ghostLabel,faceCells[1],&ghost);CHKERRQ(ierr);
//        if (ghost <= 0) {ierr = DMPlexPointLocalFieldRef(dm, faceCells[1], f, fa, &fR);CHKERRQ(ierr);}
//
//        if(fL){
//            fL[RHOE] -= normalFlux/cgL->volume;
//        }
//        if(fR){
//            fR[RHOE] += normalFlux/cgR->volume;
//        }
//    }
//
//    // Add the new locFVec to the globFVec
//    ierr = VecRestoreArray(locFVec, &fa);CHKERRQ(ierr);
//    ierr = DMLocalToGlobalBegin(dm, locFVec, INSERT_VALUES, globFVec);CHKERRQ(ierr);
//    ierr = DMLocalToGlobalEnd(dm, locFVec, INSERT_VALUES, globFVec);CHKERRQ(ierr);
//    ierr = DMRestoreLocalVector(dm, &locFVec);CHKERRQ(ierr);
//
//    // restore the arrays
//    ierr = VecRestoreArrayRead(cellGeomVec, &cellGeomArray);CHKERRQ(ierr);
//    ierr = VecRestoreArrayRead(faceGeomVec, &faceGeomArray);CHKERRQ(ierr);
//    ierr = VecRestoreArrayRead(locXVec, &locXArray);CHKERRQ(ierr);
//
//    PetscFunctionReturn(0);
//}
//
//static PetscErrorCode DMPlexInsertBoundaryGradientValuesRiemann(DM dm, PetscReal time, Vec faceGeometry, Vec cellGeometry, PetscInt field, PetscInt Nc, const PetscInt comps[], DMLabel label, PetscInt numids, const PetscInt ids[],
//                                                 PetscErrorCode (*func)(PetscReal,const PetscReal*,const PetscReal*,const PetscScalar*,PetscScalar*,void*), void *ctx, Vec locGradXVec)
//{
//    PetscDS            prob;
//    PetscSF            sf;
//    DM                 dmFace, dmCell;
//    const PetscScalar *facegeom, *cellgeom = NULL;
//    const PetscInt    *leaves;
//    PetscScalar       *locGradXArray;
//    PetscInt           dim, nleaves, loc, fStart, fEnd, pdim, i;
//    PetscErrorCode     ierr, ierru = 0;
//
//    PetscFunctionBegin;
//    ierr = DMGetPointSF(dm, &sf);CHKERRQ(ierr);
//    ierr = PetscSFGetGraph(sf, NULL, &nleaves, &leaves, NULL);CHKERRQ(ierr);
//    nleaves = PetscMax(0, nleaves);
//    ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
//    ierr = DMPlexGetHeightStratum(dm, 1, &fStart, &fEnd);CHKERRQ(ierr);
//    ierr = DMGetDS(dm, &prob);CHKERRQ(ierr);
//    ierr = VecGetDM(faceGeometry, &dmFace);CHKERRQ(ierr);
//    ierr = VecGetArrayRead(faceGeometry, &facegeom);CHKERRQ(ierr);
//    if (cellGeometry) {
//        ierr = VecGetDM(cellGeometry, &dmCell);CHKERRQ(ierr);
//        ierr = VecGetArrayRead(cellGeometry, &cellgeom);CHKERRQ(ierr);
//    }
//
//    ierr = VecGetArray(locGradXVec, &locGradXArray);CHKERRQ(ierr);
//    for (i = 0; i < numids; ++i) {
//        IS              faceIS;
//        const PetscInt *faces;
//        PetscInt        numFaces, f;
//
//        ierr = DMLabelGetStratumIS(label, ids[i], &faceIS);CHKERRQ(ierr);
//        if (!faceIS) continue; /* No points with that id on this process */
//        ierr = ISGetLocalSize(faceIS, &numFaces);CHKERRQ(ierr);
//        ierr = ISGetIndices(faceIS, &faces);CHKERRQ(ierr);
//        for (f = 0; f < numFaces; ++f) {
//            const PetscInt         face = faces[f], *cells;
//            PetscFVFaceGeom        *fg;
//
//            if ((face < fStart) || (face >= fEnd)) continue; /* Refinement adds non-faces to labels */
//            ierr = PetscFindInt(face, nleaves, (PetscInt *) leaves, &loc);CHKERRQ(ierr);
//            if (loc >= 0) continue;
//            ierr = DMPlexPointLocalRead(dmFace, face, facegeom, &fg);CHKERRQ(ierr);
//            ierr = DMPlexGetSupport(dm, face, &cells);CHKERRQ(ierr);
//
//                PetscScalar       *xI;
//                PetscScalar       *xG;
//
//                ierr = DMPlexPointLocalRead(dm, cells[0], locGradXArray, &xI);CHKERRQ(ierr);
//                ierr = DMPlexPointLocalFieldRef(dm, cells[1], field, locGradXArray, &xG);CHKERRQ(ierr);
//                ierru = (*func)(time, fg->centroid, fg->normal, xI, xG, ctx);
//                if (ierru) {
//                    ierr = ISRestoreIndices(faceIS, &faces);
//                    CHKERRQ(ierr);
//                    ierr = ISDestroy(&faceIS);
//                    CHKERRQ(ierr);
//                    goto cleanup;
//                }
//        }
//        ierr = ISRestoreIndices(faceIS, &faces);CHKERRQ(ierr);
//        ierr = ISDestroy(&faceIS);CHKERRQ(ierr);
//    }
//    cleanup:
//    ierr = VecRestoreArray(locGradXVec, &locGradXArray);CHKERRQ(ierr);
//    if (cellGeometry) {ierr = VecRestoreArrayRead(cellGeometry, &cellgeom);CHKERRQ(ierr);}
//    ierr = VecRestoreArrayRead(faceGeometry, &facegeom);CHKERRQ(ierr);
//    CHKERRQ(ierru);
//    PetscFunctionReturn(0);
//}

static PetscErrorCode FillBoundary(PetscInt dim, PetscInt dof, const PetscFVFaceGeom *faceGeom, const PetscFVCellGeom *cellGeom, const PetscFVCellGeom *cellGeomG, const PetscScalar *a_xI, const PetscScalar *a_xGradI, const PetscScalar *a_xG,  PetscScalar *a_xGradG, void *ctx){

    for(PetscInt pd = 0; pd < dof; ++pd) {
        PetscReal dPhidS = a_xG[pd] - a_xI[pd];

        // over each direction
        for (PetscInt dir = 0; dir < dim; dir++) {
            PetscReal dx = (cellGeomG->centroid[dir] - faceGeom->centroid[dir]);

            // If there is a contribution in this direction
            if (PetscAbs(dx) > 1E-8) {
                a_xGradG[pd*dim + dir] = dPhidS / (dx);
            } else {
                a_xGradG[pd*dim + dir] = 0.0;
            }
        }
    }
    return 0;
}

/**
 * this function updates the boundaries with the gradient computed from the boundary cell value
 * @param dm
 * @param auxFvm
 * @param gradLocalVec
 * @return
 */
static PetscErrorCode DMPlexFillGradientInBoundaryFVM(DM dm, PetscFV auxFvm, Vec localXVec, Vec gradLocalVec){
    PetscFunctionBeginUser;
    PetscErrorCode ierr;

    // Get the dmGrad
    DM dmGrad;
    ierr = VecGetDM(gradLocalVec, &dmGrad);CHKERRQ(ierr);

    // get the problem
    PetscDS prob;
    PetscInt nFields;
    ierr = DMGetDS(dm, &prob);CHKERRQ(ierr);
    ierr = PetscDSGetNumFields(prob, &nFields);CHKERRQ(ierr);
    if (nFields > 1){
        SETERRQ(PetscObjectComm((PetscObject) dm), PETSC_ERR_ARG_WRONG, "Cannot fill DMPlexFillGradientInBoundaryFVM with more than a single field." );
    }
    PetscInt field;
    ierr = PetscDSGetFieldIndex(prob, (PetscObject) auxFvm, &field);CHKERRQ(ierr);
    PetscInt dof;
    ierr = PetscDSGetFieldSize(prob, field, &dof);CHKERRQ(ierr);

    // Obtaining local cell ownership
    PetscInt cellStart, cellEnd, cEndInterior;
    ierr = DMPlexGetHeightStratum(dm, 0, &cellStart, &cellEnd);CHKERRQ(ierr);
    ierr = DMPlexGetGhostCellStratum(dm, &cEndInterior, NULL);CHKERRQ(ierr);
    cEndInterior = cEndInterior < 0 ? cellEnd: cEndInterior;

    // Get the fvm face and cell geometry
    Vec cellGeomVec = NULL;/* vector of structs related to cell geometry*/
    Vec faceGeomVec = NULL;/* vector of structs related to face geometry*/
    ierr = DMPlexGetGeometryFVM(dm, &faceGeomVec, &cellGeomVec, NULL);CHKERRQ(ierr);

    // get the dm for each geom type
    DM dmFaceGeom, dmCellGeom;
    ierr = VecGetDM(faceGeomVec, &dmFaceGeom);CHKERRQ(ierr);
    ierr = VecGetDM(cellGeomVec, &dmCellGeom);CHKERRQ(ierr);

    // extract the gradLocalVec
    PetscScalar * gradLocalArray;
    ierr = VecGetArray(gradLocalVec, &gradLocalArray);CHKERRQ(ierr);

    const PetscScalar* localArray;
    ierr = VecGetArrayRead(localXVec, &localArray);CHKERRQ(ierr);

    // get the dim
    PetscInt dim;
    ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);

    // extract the arrays for the face and cell geom, along with their dm
    const PetscScalar *cellGeomArray;
    ierr = VecGetArrayRead(cellGeomVec, &cellGeomArray);CHKERRQ(ierr);

    const PetscScalar *faceGeomArray;
    ierr = VecGetArrayRead(faceGeomVec, &faceGeomArray);CHKERRQ(ierr);

    // March over each face
    for (PetscInt cell = cellStart; cell < cEndInterior; ++cell) {
        // Get the face information
        PetscInt               numFaces;
        const PetscInt        *faces;
        ierr = DMPlexGetConeSize(dm, cell, &numFaces);CHKERRQ(ierr);
        ierr = DMPlexGetCone(dm, cell, &faces);CHKERRQ(ierr);

        //March over each face
        for(PetscInt f =0 ; f < numFaces; f++){
            PetscBool boundary;
            ierr = DMIsBoundaryPoint(dm, faces[f], &boundary);CHKERRQ(ierr);

            // if this is on the boundary
            if(boundary){
                // get the boundary cell index
                const PetscInt        *fcells;
                ierr  = DMPlexGetSupport(dm, faces[f], &fcells);CHKERRQ(ierr);
                PetscInt side  = (cell != fcells[0]); /* c is on left=0 or right=1 of face */
                PetscInt ncell = fcells[!side];    /* the neighbor */

                // get the face geom
                const PetscFVFaceGeom  *faceGeom;
                ierr  = DMPlexPointLocalRead(dmFaceGeom, faces[f], faceGeomArray, &faceGeom);CHKERRQ(ierr);

                // get the cell centroid information
                const PetscFVCellGeom       *cellGeom;
                const PetscFVCellGeom       *cellGeomGhost;
                ierr  = DMPlexPointLocalRead(dmCellGeom, cell, cellGeomArray, &cellGeom);CHKERRQ(ierr);
                ierr  = DMPlexPointLocalRead(dmCellGeom, ncell, cellGeomArray, &cellGeomGhost);CHKERRQ(ierr);

                // Read the local point
                PetscScalar* boundaryGradCellValues;
                ierr  = DMPlexPointLocalRef(dmGrad, ncell, gradLocalArray, &boundaryGradCellValues);CHKERRQ(ierr);

                const PetscScalar*  cellGradValues;
                ierr  = DMPlexPointLocalRead(dmGrad, cell, gradLocalArray, &cellGradValues);CHKERRQ(ierr);

                const PetscScalar* boundaryCellValues;
                ierr  = DMPlexPointLocalRead(dm, ncell, localArray, &boundaryCellValues);CHKERRQ(ierr);
                const PetscScalar* cellValues;
                ierr  = DMPlexPointLocalRead(dm, cell, localArray, &cellValues);CHKERRQ(ierr);

                // compute the gradient for the boundary node and pass in
                //const PetscReal *cellGeom, const PetscReal *cellGeomG, const PetscScalar *a_xI, const PetscScalar *a_xGradI, PetscScalar *a_xG,  PetscScalar *a_xGradG, void *ctx
                ierr = FillBoundary(dim, dof, faceGeom, cellGeom, cellGeomGhost, cellValues, cellGradValues, boundaryCellValues, boundaryGradCellValues, NULL);CHKERRQ(ierr);
            }
        }
    }

    ierr = VecRestoreArrayRead(localXVec, &localArray);CHKERRQ(ierr);
    ierr = VecRestoreArray(gradLocalVec, &gradLocalArray);CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(cellGeomVec, &cellGeomArray);CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(faceGeomVec, &faceGeomArray);CHKERRQ(ierr);

    PetscFunctionReturn(0);
}

static PetscErrorCode DiffusionSource(DM dm, PetscReal time, Vec locXVec, Vec globFVec, void *ctx){
    // Call the flux calculation
    PetscErrorCode ierr;

    ProblemSetup *setup = (ProblemSetup *)ctx;

    // call the base rhs function eval
    ierr = DMPlexTSComputeRHSFunctionFVM(dm, time, locXVec, globFVec, setup->flowData);CHKERRQ(ierr);

    // update the aux fields
    ierr = UpdateAuxFields(NULL, locXVec, setup->flowData);CHKERRQ(ierr);

    Constants constants = setup->constants;

    // get the fvm field assuming that it is the first
    PetscFV auxFvm;
    ierr = DMGetField(setup->flowData->auxDm,0, NULL, (PetscObject*)&auxFvm);CHKERRQ(ierr);

    PetscInt dim;
    ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);

    // Get the locXArray
    const PetscScalar *locXArray;
    ierr = VecGetArrayRead(locXVec, &locXArray);CHKERRQ(ierr);

    // Get the fvm face and cell geometry
    Vec cellGeomVec = NULL;/* vector of structs related to cell geometry*/
    Vec faceGeomVec = NULL;/* vector of structs related to face geometry*/

    // extract the fvm data
    ierr = DMPlexGetGeometryFVM(setup->flowData->dm, &faceGeomVec, &cellGeomVec, NULL);CHKERRQ(ierr);

    // Get the needed auxDm
    DM auxFieldGradDM = NULL; /* dm holding the grad information */
    ierr = DMPlexGetDataFVM(setup->flowData->auxDm, auxFvm, NULL, NULL, &auxFieldGradDM);CHKERRQ(ierr);
    if(!auxFieldGradDM){
        SETERRQ(PetscObjectComm((PetscObject)setup->flowData->auxDm), PETSC_ERR_ARG_WRONGSTATE, "The FVM method for aux variables must support computing gradients.");
    }

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

    // get the ghost label
    DMLabel ghostLabel;
    ierr = DMGetLabel(dm, "ghost", &ghostLabel);CHKERRQ(ierr);

    // extract the localFArray from the locFVec
    PetscScalar *fa;
    Vec locFVec;
    ierr = DMGetLocalVector(dm, &locFVec);CHKERRQ(ierr);
    ierr = VecZeroEntries(locFVec);CHKERRQ(ierr);
    ierr = VecGetArray(locFVec, &fa);CHKERRQ(ierr);

    // create a global and local grad vector for the auxField
    Vec gradGlobalVec, gradLocalVec;
    ierr = DMCreateGlobalVector(auxFieldGradDM, &gradGlobalVec);CHKERRQ(ierr);
    ierr = VecSet(gradGlobalVec, NAN);CHKERRQ(ierr);

    // compute the global grad values
    ierr = DMPlexReconstructGradientsFVM(setup->flowData->auxDm, setup->flowData->auxField, gradGlobalVec);CHKERRQ(ierr);

    // Map to a local grad vector
    ierr = DMCreateLocalVector(auxFieldGradDM, &gradLocalVec);CHKERRQ(ierr);
    ierr = DMGlobalToLocalBegin(auxFieldGradDM, gradGlobalVec, INSERT_VALUES, gradLocalVec);CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(auxFieldGradDM, gradGlobalVec, INSERT_VALUES, gradLocalVec);CHKERRQ(ierr);

    // fill the boundary conditions
    ierr = DMPlexFillGradientInBoundaryFVM(setup->flowData->auxDm, auxFvm, setup->flowData->auxField,  gradLocalVec);CHKERRQ(ierr);

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
        ierr = DMPlexPointLocalRead(auxFieldGradDM, faceCells[0], localGradArray, &gradL);CHKERRQ(ierr);
        ierr = DMPlexPointLocalRead(auxFieldGradDM, faceCells[1], localGradArray, &gradR);CHKERRQ(ierr);

        PetscInt f = 0;

        // extract the field values
        PetscScalar *xL, *xR,
        ierr = DMPlexPointLocalFieldRead(dm, faceCells[0], f, locXArray, &xL);CHKERRQ(ierr);
        ierr = DMPlexPointLocalFieldRead(dm, faceCells[1], f, locXArray, &xR);CHKERRQ(ierr);

        // Compute the normal grad
        PetscReal normalGrad = 0.0;
        PetscInt dof = 0;
        for (PetscInt d = 0; d < dim; ++d){
            normalGrad += fg->normal[d]*0.5*(gradL[dof*dim + d] + gradR[dof*dim + d]);
        }

        // compute the normal heatFlux
        PetscReal normalHeatFlux = -constants.k *normalGrad;

        // Add to the source terms of f
        PetscScalar    *fL = NULL, *fR = NULL;
        ierr = DMLabelGetValue(ghostLabel,faceCells[0],&ghost);CHKERRQ(ierr);
        if (ghost <= 0) {ierr = DMPlexPointLocalFieldRef(dm, faceCells[0], f, fa, &fL);CHKERRQ(ierr);}
        ierr = DMLabelGetValue(ghostLabel,faceCells[1],&ghost);CHKERRQ(ierr);
        if (ghost <= 0) {ierr = DMPlexPointLocalFieldRef(dm, faceCells[1], f, fa, &fR);CHKERRQ(ierr);}

        if(fL){
            fL[RHOE] -= normalHeatFlux/cgL->volume;
        }
        if(fR){
            fR[RHOE] += normalHeatFlux/cgR->volume;
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

//    VecView(globFVec, PETSC_VIEWER_STDOUT_WORLD);

    PetscFunctionReturn(0);
}

static PetscErrorCode PhysicsBoundary_Euler(PetscReal time, const PetscReal *c, const PetscReal *n, const PetscScalar *a_xI, PetscScalar *a_xG, void *ctx) {
    PetscFunctionBeginUser;
    Constants *constants = (Constants *)ctx;

    // compute the centroid location of the real cell
    // Offset the calc assuming the cells are square
    PetscReal x[3];
    for(PetscInt i =0; i < constants->dim; i++){
//        x[i] = c[i] - n[i]*0.5;//TODO:zero boundary
        x[i] = c[i] + n[i]*0.5;//TODO:uniform grad
    }

    // compute the temperature
    PetscReal Tinside =   ComputeTExact(time, x, constants, 1.0);
    PetscReal boundaryValue = 300.0;

    PetscReal T = boundaryValue;//boundaryValue - (Tinside - boundaryValue);//TODO: fix uniform grad

//    PetscReal T = c[0] < constants->L/2.0 ? 300 : 400;

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
    constants.L = 0.2;
    constants.gamma = 1.4;
    constants.k = 0.3;
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
    ierr = TSSetType(ts, TSEULER);CHKERRQ(ierr);
    ierr = TSSetExactFinalTime(ts, TS_EXACTFINALTIME_MATCHSTEP);CHKERRQ(ierr);

    // Create a mesh
    // hard code the problem setup
    PetscReal start[] = {0.0, 0.0};
    PetscReal end[] = {constants.L, constants.L};
    PetscReal nxHeight = 3;
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
    params[RGAS] = constants.Rgas;

    // set up the finite volume fluxes
    CompressibleFlow_StartProblemSetup(flowData, TOTAL_COMPRESSIBLE_FLOW_PARAMETERS, params);

    // Add in any boundary conditions
    PetscDS prob;
    ierr = DMGetDS(flowData->dm, &prob);
    CHKERRABORT(PETSC_COMM_WORLD, ierr);
    const PetscInt idsLeft[]= {2, 4};
    ierr = PetscDSAddBoundary(prob, DM_BC_NATURAL_RIEMANN, "wall left", "Face Sets", 0, 0, NULL, (void (*)(void))PhysicsBoundary_Euler, NULL, 2, idsLeft, &constants);CHKERRQ(ierr);

    const PetscInt idsTop[]= {1, 3};
    ierr = PetscDSAddBoundary(prob, DM_BC_NATURAL_RIEMANN, "top/bottom", "Face Sets", 0, 0, NULL, (void (*)(void))PhysicsBoundary_Mirror, NULL, 2, idsTop, &constants);CHKERRQ(ierr);

//    ierr = FlowRegisterAuxField(flowData, "Temperature", "T", 1, FV);CHKERRQ(ierr);

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
    PetscReal cv = constants.gamma*constants.Rgas/(constants.gamma - 1) - constants.Rgas;
    PetscReal dxMin = constants.L/(10 * PetscPowRealInt(2, 2));

    PetscReal alpha = PetscAbs(constants.k/ (cv * 1.0));
    double dt_conduc =  5.000000000000004E-4;// 0.00000625;//.3*PetscSqr(dxMin) / alpha;

    PetscPrintf(PETSC_COMM_WORLD, "dt_conduc: %f\n", dt_conduc);
    TSSetTimeStep(ts, dt_conduc);
//    TSSetMaxTime(ts, 0.001);
    TSSetMaxSteps(ts, 1200);

    PetscDSView(prob, PETSC_VIEWER_STDOUT_WORLD);

    // Add in any boundary conditions
    PetscDS auxProblem;
    ierr = DMGetDS(flowData->auxDm, &auxProblem);CHKERRQ(ierr);


    DMView(flowData->auxDm, PETSC_VIEWER_STDOUT_WORLD);

    ierr = TSSolve(ts,flowData->flowField);CHKERRQ(ierr);

    FlowDestroy(&flowData);
    TSDestroy(&ts);

    return PetscFinalize();

}