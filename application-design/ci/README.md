## Continuous integration with Github Actions

In this exercise you can familiarize yourself with Github Actions.

1. Modify [mpi-test.yml](../../.github/workflows/mpi-test.yml 
../../.github/workflows/mpi-test.yml) so that it is triggered by **push** events.
2. Check from Github web-GUI (Actions) if the workflow is run, and whether it is succesful.
3. Try to add a new "Step" which build and tests some of your own solutions. Note that
   Github does not provide GPU resources, so only CPU codes can be tested.
