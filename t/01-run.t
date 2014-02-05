use Test::More 'no_plan';
use strict;
use File::Temp 'tempdir';
use AI::FANN::Evolving;
use Algorithm::Genetic::Diploid::Logger ':levels';

# instantiate factory
my $fac = AI::FANN::Evolving::Factory->new;
ok( $fac->can('create') );

# create the training data for a XOR operator:
my $xor_train = AI::FANN::TrainData->new( 
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
	'env'       => $xor_train,
);
ok( $exp->isa('Algorithm::Genetic::Diploid::Experiment') );

# initialize the experiment
ok( $exp->initialize );

$exp->logger->level( 'level' => INFO );

# run!
$exp->run();