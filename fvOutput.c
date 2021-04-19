#include <petsc.h>

static PetscErrorCode SetInitialCondition(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar *u, void *ctx){

    u[0] = x[0];
    return 0;
}

int main(int argc, char **argv)
{
    PetscErrorCode ierr;
    // create the mesh
    // setup the ts
    DM dm;                 /* problem definition */

    // initialize petsc and mpi
    PetscInitialize(&argc, &argv, NULL, "HELP");


    //PetscErrorCode DMPlexCreateBoxMesh(MPI_Comm comm, PetscInt dim, PetscBool simplex, const PetscInt faces[], const PetscReal lower[], const PetscReal upper[], const DMBoundaryType periodicity[], PetscBool interpolate, DM *dm)
    PetscReal start[] = {-1, -1};
    PetscReal end[] = {1.0, 1.0};
    PetscInt nx[] = {10, 10};
    DMBoundaryType bcType[] = {DM_BOUNDARY_NONE, DM_BOUNDARY_NONE};
    ierr = DMPlexCreateBoxMesh(PETSC_COMM_WORLD, 2, PETSC_FALSE, nx, start, end, bcType, PETSC_TRUE, &dm);CHKERRQ(ierr);

    ierr = DMViewFromOptions(dm, NULL, "-dm_view");CHKERRQ(ierr);

    {
        DM dmDist;

//        ierr = DMSetBasicAdjacency(dm, PETSC_TRUE, PETSC_FALSE);CHKERRQ(ierr);
        ierr = DMPlexDistribute(dm, 1, NULL, &dmDist);CHKERRQ(ierr);
        if (dmDist) {
            ierr = DMDestroy(&dm);CHKERRQ(ierr);
            dm   = dmDist;
        }
    }

    ierr = DMSetFromOptions(dm);CHKERRQ(ierr);

    {
        DM gdm;
        ierr = DMPlexConstructGhostCells(dm, NULL, NULL, &gdm);CHKERRQ(ierr);
        ierr = DMDestroy(&dm);CHKERRQ(ierr);
        dm   = gdm;
    }

    ierr = DMViewFromOptions(dm, NULL, "-dm_view_ghost");CHKERRQ(ierr);


    // setup the FV field
    PetscFV           fvm;
    ierr = PetscFVCreate(PETSC_COMM_WORLD, &fvm);CHKERRQ(ierr);

    ierr = PetscFVSetFromOptions(fvm);CHKERRQ(ierr);
    ierr = PetscFVSetNumComponents(fvm, 1);CHKERRQ(ierr);
    ierr = PetscFVSetSpatialDimension(fvm, 2);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject) fvm,"outputField");CHKERRQ(ierr);

    /* FV is now structured with one field having all physics as components */
    ierr = DMAddField(dm, NULL, (PetscObject) fvm);CHKERRQ(ierr);

    PetscDS           prob;
    ierr = DMCreateDS(dm);CHKERRQ(ierr);
    ierr = DMGetDS(dm, &prob);CHKERRQ(ierr);

    // setup the solution vector, this olds everything
    Vec X;
    ierr = DMCreateGlobalVector(dm, &X);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject) X, "solution");CHKERRQ(ierr);

    // set the initial conditions
    PetscErrorCode     (*func[1]) (PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar *u, void *ctx) = {SetInitialCondition};
    ierr    = DMProjectFunction(dm,0.0,func,NULL,INSERT_ALL_VALUES,X);CHKERRQ(ierr);

    ierr = VecViewFromOptions(X, NULL, "-vec_view");CHKERRQ(ierr);

    ierr = VecDestroy(&X);CHKERRQ(ierr);
    ierr = DMDestroy(&dm);CHKERRQ(ierr);

    return PetscFinalize();

}