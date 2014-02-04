package AI::FANN::Evolving::Gene;
use strict;
use warnings;
use List::Util 'shuffle';
use Scalar::Util 'refaddr';
use AI::FANN::Evolving;
use AI::FANN::Evolving::TrainData;
use Algorithm::Genetic::Diploid;
use base 'Algorithm::Genetic::Diploid::Gene';

my $log = __PACKAGE__->logger;

sub new {

	# initialize self up the inheritance tree
	my $self = shift->SUPER::new(@_);
			
	# instantiate and train the FANN object
	my $traindata = $self->experiment->traindata;
	$self->ann( AI::FANN::Evolving->new( 'data' => $traindata ) );
	return $self;
}

sub ann {
	my $self = shift;
	$self->{'ann'} = shift if @_;
	return $self->{'ann'};
}

# returns a code reference to the fitness function
sub make_function {
	my $self = shift;
	$log->debug("making fitness function");
	
	# build the fitness function
	return sub {
	
		# isa TrainingData object, this is what we need to use
		# to make our prognostications. It is a different data 
		# set (out of sample) than the TrainingData object that
		# the AI was trained on.
		my $env = shift;
		my @expected = $env->get_dependent;
		my $neurons = $self->neurons;		
		$log->debug("adaptive landscape is $env");

		# test all predictions
		my ( $lls, $ls ) = ( 0, 0 ); # Longest Losing Streak
		my ( $right, $wrong ) = ( 0, 0 );
		my $ann = $self->ann;
		for my $i ( 0 .. $#expected - $neurons - 1 ) {
			
			# run prognostication over a time window
			my @window;
			for my $j ( $i .. $i + $neurons ) {
				push @window, $expected[$j];
			}
			my $obs = $ann->run( \@window )->[0];
			
			# get the expectation, this is one day beyond the window
			my $exp = $expected[ $i + $neurons + 1 ];
			
			# prediction is wrong if $exp and $obs
			# are on either side of zero
			if ( $exp > 0 xor $obs > 0 ) {
				$wrong++;
				$ls++;
				$lls = $ls if $ls > $lls;
			}
			else {
				$right++;
				$ls = 0;
			}

			$log->debug( "expected: $exp - observed: $obs" );
		}
		
		# return value is $wrong / $right, so optimum is 0
		my $fitness = $wrong / $right;
		$self->{'fitness'} = $fitness;
		my $id = $self->id;
		$log->info("fitness = $fitness, longest losing streak = $lls, id = $id");
		$self->{'fann_file'} = $self->experiment->workdir . "/${fitness}.ann";
		$self->ann->save($self->{'fann_file'});
		return $self->{'fitness'};
	}
}

sub fitness { shift->{'fitness'} }

sub mutate {
	my $self = shift;
	
	# probably 0.05
	my $mu = $self->experiment->mutation_rate;

	# make a clone, which we might mutate further
	my $ann_clone = $self->ann->clone;
	$self = $self->clone;
	$self->ann( $ann_clone );
	
	# properties of ann we might mutate
	for my $prop ( AI::FANN::Evolving->continuous_properties ) {
	
		# mutate by a value <= $mu
		my $change = rand($mu);
		$log->debug("going to mutate $prop by $change");
		my $propval  = $ann_clone->$prop;
		$propval = $mu if $propval == 0;
		my $npropval = $propval + ( $propval * $change - $propval * $mu / 2 );
		
		# we don't want the sign to change, e.g. a change from 0.01 to -0.01 might
		# have unforeseen effects, so flip the sign if we cross zero.
		$npropval *= -1 if $propval < 0 xor $npropval < 0;
		$ann_clone->$prop( $npropval );
	}
	$self->experiment->fann_trainer( 
		$ann_clone, $self->make_timeseries_data, $self->neurons
	);
	return $self;
}

1;