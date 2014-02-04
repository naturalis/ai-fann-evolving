package AI::FANN::Evolving;
use strict;
use AI::FANN ':all';
use File::Temp 'tempfile';
use Scalar::Util 'refaddr';
use AI::FANN::Evolving::Gene;
use AI::FANN::Evolving::Chromosome;
use AI::FANN::Evolving::Experiment;
use Algorithm::Genetic::Diploid;
use base 'AI::FANN';

our $VERSION = '0.1';
my $log = Algorithm::Genetic::Diploid->logger;
my %self;

# requires 'file', or 'data' and 'neurons' arguments. optionally takes 
# 'connection_rate' argument for sparse topologies. all other arguments
# are handled by _init
sub new {
	my $class = shift;
	my %args  = @_;
	
	# de-serialize from a file
	if ( my $file = $args{'file'} ) {
		my $self = $class->new_from_file($file);
		return $self->_init(%args);
	}
	
	# build new topology
	elsif ( my $data = $args{'data'} ) {
		die "Need 'neurons' argument when building from scratch!";
		my @sizes = ( 
			$data->num_inputs, 
			$args{'neurons'},
			$data->num_outputs
		);
		
		# build sparse topology
		my $self;
		if ( $args{'connection_rate'} ) {
			$self = $class->new_sparse( $args{'connection_rate'}, @sizes );
		}
		else {
			$self = $class->new_standard( @sizes );
		}
		
		# finalize the instance
		$self->_init;
		$self->train( $data );
		return $self;
	}
	else {
		die "Need 'file' or 'data' argument!";
	}
}

sub _init {
	my $self = shift;
	my $id   = refaddr $self;
	my %args = @_;
	$self{$id} = {
		'error'               => $args{'error'}               || 0.0001,
		'epochs'              => $args{'epochs'}              || 5000,		
		'training_type'       => $args{'training_type'}       || 'ordinary',		
		'epoch_printfreq'     => $args{'epoch_printfreq'}     || 1000,
		'neuron_printfreq'    => $args{'neuron_printfreq'}    || 10,
		'neurons'             => $args{'neurons'}             || 15,
		'activation_function' => $args{'activation_function'} || FANN_SIGMOID_SYMMETRIC,		
	};	
	return $self;
}

sub DESTROY {
	my $self = shift;
	my $id   = refaddr $self;
	delete $self{$id};	
}

sub clone {
	my $self = shift;
	my ( $fh, $name ) = tempfile();
	$self->save($name);
	my $clone = __PACKAGE__->new( 'file' => $name );
	unlink $name;
	return $clone;
}

# trains the AI on the provided data object
sub train {
	my ( $self, $data ) = @_;
	if ( $self->train_type eq 'cascade' ) {
	
		# set learning curve
		$self->cascade_activation_functions( $self->activation_function );
		
		# train
		$self->cascadetrain_on_data(
			$data,
			$self->neurons,
			$self->neuron_printfreq,
			$self->error,
		);
	}
	else {
	
		# set learning curves
		$self->hidden_activation_function( $self->activation_function );
		$self->output_activation_function( $self->activation_function );
		
		# train
		$self->train_on_data(
			$data,
			$self->epochs,
			$self->epoch_printfreq,
			$self->error,
		);	
	}
}

# returns a list of names for the AI::FANN properties that are
# continuous valued and that could be evolved
sub continuous_properties {
	(
		'learning_rate',
#		'learning_momentum',
		'quickprop_decay',
		'quickprop_mu',
		'rprop_increase_factor',
		'rprop_decrease_factor',
#		'cascade_output_change_fraction',
		'cascade_candidate_change_fraction',
		'cascade_candidate_limit',
		'cascade_weight_multiplier',
#		'rprop_delta_min',
#		'rprop_delta_max',
	);
}

# returns a list of names for the AI::FANN properties that are
# discrete valued and that could be evolved
sub discrete_properties {

}

# default is 0.0001
sub error {
	my $ref = shift;
	my $id = refaddr $ref;
	$self{$id}->{'error'} = shift if @_;
	return $self{$id}->{'error'};
}

# number of training epochs, default is 500000
sub epochs {
	my $ref = shift;
	my $id = refaddr $ref;
	$self{$id}->{'epochs'} = shift if @_;
	return $self{$id}->{'epochs'};
}

# number of epochs after which progress is printed. default is 1000
sub epoch_printfreq {
	my $ref = shift;
	my $id = refaddr $ref;
	$self{$id}->{'epoch_printfreq'} = shift if @_;
	return $self{$id}->{'epoch_printfreq'};
}

# number of neurons
sub neurons {
	my $ref = shift;
	my $id = refaddr $ref;
	$self{$id}->{'neurons'} = shift if @_;
	return $self{$id}->{'neurons'};
}

# number of cascading neurons after which progress is printed. default is 10
sub neuron_printfreq {
	my $ref = shift;
	my $id = refaddr $ref;
	$self{$id}->{'neuron_printfreq'} = shift if @_;
	return $self{$id}->{'neuron_printfreq'};
}

# 'cascade' or 'ordinary'. default is ordinary
sub train_type {
	my $ref = shift;
	my $id = refaddr $ref;
	$self{$id}->{'train_type'} = lc shift if @_;
	return $self{$id}->{'train_type'};
}

# the function that maps inputs to outputs. default is FANN_SIGMOID_SYMMETRIC
sub activation_function {
	my $ref = shift;
	my $id = refaddr $ref;
	$self{$id}->{'activation_function'} = shift if @_;
	return $self{$id}->{'activation_function'};
}

1;