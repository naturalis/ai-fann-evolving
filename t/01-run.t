use Test::More 'no_plan';
use strict;
use File::Temp 'tempdir';
use AI::FANN::Evolving;

# instantiate factory
my $fac = AI::FANN::Evolving::Factory->new;
ok( $fac->can('create') )

# create the training data for a XOR operator:
my $xor_train = $fac->create_traindata( 
	[-1, -1] => [-1],
	[-1,  1] => [ 1],
    [ 1, -1] => [ 1],
    [ 1,  1] => [-1] 
);
ok( $xor_train->isa('AI::FANN::TrainData') );

# create the experiment
my $exp = $fac->create_experiment( 
	'workdir'   => tempdir( 'CLEANUP' => 1 ),
	'traindata' => $xor_train,
	'factory'   => $fac,
);
ok( $exp->isa('Algorithm::Genetic::Diploid') );

# initialize the experiment
$exp->initialize();