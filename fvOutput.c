#include <petsc.h>

static PetscErrorCode SetInitialCondition(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar *u, void *ctx){

    u[0] = x[0];
    return 0;
}


static PetscErrorCode PrintFVGradients(DM dm){
    PetscFunctionBeginUser;
    PetscErrorCode ierr;

    // Convert the dm to a plex
    DM plex;
    DMConvert(dm, DMPLEX, &plex);

    // Get the start/end
    PetscInt fStart;
    PetscInt fEnd;
    ierr = DMPlexGetHeightStratum(plex, 1, &fStart, &fEnd);CHKERRQ(ierr);

    // Extract the cell geometry, and the dm that holds the information
    Vec faceGeometry;
    DM dmFace;
    ierr = DMPlexGetGeometryFVM(plex, &faceGeometry, NULL, NULL);CHKERRQ(ierr);

    ierr = VecGetDM(faceGeometry, &dmFace);CHKERRQ(ierr);

    const PetscScalar *facegeom;
    ierr = VecGetArrayRead(faceGeometry, &facegeom);CHKERRQ(ierr);

    // Get the dim
    PetscInt dim;
    ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);

    // March over each cell volume
    for (PetscInt f = fStart; f < fEnd; ++f) {
        PetscFVFaceGeom       *fg;

        ierr = DMPlexPointLocalRead(dmFace, f, facegeom, &fg);CHKERRQ(ierr);

        printf("Face Grad %d: ", f);
        for(PetscInt d =0; d< dim; d++){
            printf("%f/%f, ", fg->grad[0][d], fg->grad[1][d]);
        }
        printf("\n");
    }

    PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
    DM             dm;
    PetscErrorCode ierr;

    ierr = PetscInitialize(&argc, &argv, NULL, "HELP");if (ierr) return(ierr);
    ierr = DMPlexCreateBoxMesh(PETSC_COMM_WORLD, 2, PETSC_FALSE, NULL, NULL, NULL, NULL, PETSC_TRUE, &dm);CHKERRQ(ierr);
    ierr = DMViewFromOptions(dm, NULL, "-dm_view");CHKERRQ(ierr);
    ierr = DMSetFromOptions(dm);CHKERRQ(ierr);

    {
        DM gdm;

        ierr = DMPlexConstructGhostCells(dm, NULL, NULL, &gdm);CHKERRQ(ierr);
        ierr = DMDestroy(&dm);CHKERRQ(ierr);
        dm   = gdm;
    }
    ierr = DMViewFromOptions(dm, NULL, "-dm_view_ghost");CHKERRQ(ierr);

    PetscFV fvm;
    PetscDS ds;

    ierr = PetscFVCreate(PETSC_COMM_WORLD, &fvm);CHKERRQ(ierr);
    ierr = PetscFVSetFromOptions(fvm);CHKERRQ(ierr);
    ierr = PetscFVSetNumComponents(fvm, 1);CHKERRQ(ierr);
    ierr = PetscFVSetSpatialDimension(fvm, 2);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject) fvm, "outputField");CHKERRQ(ierr);

    /* FV is now structured with one field having all physics as components */
    ierr = DMAddField(dm, NULL, (PetscObject) fvm);CHKERRQ(ierr);
    ierr = PetscFVDestroy(&fvm);CHKERRQ(ierr);
    ierr = DMCreateDS(dm);CHKERRQ(ierr);
    ierr = DMGetDS(dm, &ds);CHKERRQ(ierr);

    Vec X;
    PetscErrorCode     (*func[1]) (PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar *u, void *ctx) = {SetInitialCondition};

    ierr = DMCreateGlobalVector(dm, &X);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject) X, "solution");CHKERRQ(ierr);
    ierr = DMProjectFunction(dm, 0.0, func, NULL, INSERT_ALL_VALUES, X);CHKERRQ(ierr);
    ierr = VecViewFromOptions(X, NULL, "-vec_view");CHKERRQ(ierr);

    PrintFVGradients(dm);

    ierr = VecDestroy(&X);CHKERRQ(ierr);
    ierr = DMDestroy(&dm);CHKERRQ(ierr);
    return PetscFinalize();
}
