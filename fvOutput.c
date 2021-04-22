#include <petsc.h>

static PetscErrorCode SetInitialCondition(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar *u, void *ctx){

    u[0] = x[0];
    return 0;
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

    ierr = VecDestroy(&X);CHKERRQ(ierr);
    ierr = DMDestroy(&dm);CHKERRQ(ierr);
    return PetscFinalize();
}
