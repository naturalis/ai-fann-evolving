use FindBin qw($Bin);
use Test::More 'no_plan';
use AI::FANN::Evolving::TrainData;
use Algorithm::Genetic::Diploid::Logger ':levels';
use Data::Dumper;

# instantiate a data object
my $file = "$Bin/../examples/butterbeetles.tsv";
my $data = AI::FANN::Evolving::TrainData->new( 'file' => $file );
ok( $data, "instantiate" );

# partition the data
my ( $d1, $d2 ) = $data->partition_data(0.2);
ok( $data->size == $d1->size + $d2->size, "partition" );

# pack data as FANN struct
ok( $d1->to_fann, "packed d1" );
ok( $d2->to_fann, "packed d2" );