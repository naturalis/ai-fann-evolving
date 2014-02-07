use Test::More 'no_plan';
use strict;
use FindBin qw($Bin);
use File::Temp 'tempdir';
use AI::FANN::Evolving;

# input files
my $dir = "$Bin/../examples";

# instantiate factory
my $fac = AI::FANN::Evolving::Factory->new;
ok( $fac->can('create') );

# create the experiment
my $exp = $fac->create_experiment( 
	'workdir'   => tempdir( 'CLEANUP' => 1 ),
	'traindata' => $fac->create_traindata( 'file' => "$dir/bb_train.tsv" )->to_fann,
	'factory'   => $fac,
	'env'       => $fac->create_traindata( 'file' => "$dir/bb_test.tsv" )->to_fann,
	'mutation_rate' => 0.1,
	'ngens'         => 10,
);
ok( $exp->isa('Algorithm::Genetic::Diploid::Experiment') );

# initialize the experiment
ok( $exp->initialize );

# run!
$exp->run();