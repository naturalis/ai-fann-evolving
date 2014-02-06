package AI::FANN::Evolving;
use strict;
use AI::FANN ':all';
use File::Temp 'tempfile';
use Scalar::Util 'refaddr';
use AI::FANN::Evolving::Gene;
use AI::FANN::Evolving::Chromosome;
use AI::FANN::Evolving::Experiment;
use AI::FANN::Evolving::Factory;
use Algorithm::Genetic::Diploid;
use base 'AI::FANN';

our $VERSION = '0.1';
my $log = Algorithm::Genetic::Diploid->logger;
my %self;

=head1 NAME

AI::FANN::Evolving - artificial neural network that evolves

=head1 METHODS

=over

=item new

Constructor requires 'file', or 'data' and 'neurons' arguments. Optionally takes 
'connection_rate' argument for sparse topologies. Returns a subclass of L<AI::FANN>.

=cut

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
		my $neurons = $args{'neurons'} || ( $data->num_inputs + 1 );
		my @sizes = ( 
			$data->num_inputs, 
			$neurons,
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
		'train_type'          => $args{'train_type'}          || 'ordinary',		
		'epoch_printfreq'     => $args{'epoch_printfreq'}     || 0,
		'neuron_printfreq'    => $args{'neuron_printfreq'}    || 0,
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

=item clone

Clones the ANN by serializing it to a temporary file and creating a new instance
from that file

=cut

sub clone {
	my $self = shift;
	my ( $fh, $name ) = tempfile( 'DIR' => AI::FANN::Evolving::Experiment->new->workdir );
	$self->save($name);
	my $clone = __PACKAGE__->new( 'file' => $name );
	unlink $name;
	return $clone;
}

=item train

Trains the AI on the provided data object

=cut

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

=item continuous_properties

Returns a list of names for the AI::FANN properties that are continuous valued and that 
can be mutated

=cut

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

=item discrete_properties

Returns a list of names for the AI::FANN properties that are discrete valued and that 
can be mutated

=cut

sub discrete_properties {

}

=item error

Getter/setter for the error rate. Default is 0.0001

=cut

sub error {
	my $ref = shift;
	my $id = refaddr $ref;
	$self{$id}->{'error'} = shift if @_;
	return $self{$id}->{'error'};
}

=item epochs

Getter/setter for the number of training epochs, default is 500000

=cut

sub epochs {
	my $ref = shift;
	my $id = refaddr $ref;
	$self{$id}->{'epochs'} = shift if @_;
	return $self{$id}->{'epochs'};
}

=item epoch_printfreq

Getter/setter for the number of epochs after which progress is printed. default is 1000

=cut

sub epoch_printfreq {
	my $ref = shift;
	my $id = refaddr $ref;
	$self{$id}->{'epoch_printfreq'} = shift if @_;
	return $self{$id}->{'epoch_printfreq'};
}

=item neurons

Getter/setter for the number of neurons. Default is 15

=cut

sub neurons {
	my $ref = shift;
	my $id = refaddr $ref;
	$self{$id}->{'neurons'} = shift if @_;
	return $self{$id}->{'neurons'};
}

=item neuron_printfreq

Getter/setter for the number of cascading neurons after which progress is printed. 
default is 10

=cut

sub neuron_printfreq {
	my $ref = shift;
	my $id = refaddr $ref;
	$self{$id}->{'neuron_printfreq'} = shift if @_;
	return $self{$id}->{'neuron_printfreq'};
}

=item train_type

Getter/setter for the training type: 'cascade' or 'ordinary'. Default is ordinary

=cut

sub train_type {
	my $ref = shift;
	my $id = refaddr $ref;
	$self{$id}->{'train_type'} = lc shift if @_;
	return $self{$id}->{'train_type'};
}

=item activation_function

Getter/setter for the function that maps inputs to outputs. default is 
FANN_SIGMOID_SYMMETRIC

=back

=cut

sub activation_function {
	my $ref = shift;
	my $id = refaddr $ref;
	$self{$id}->{'activation_function'} = shift if @_;
	return $self{$id}->{'activation_function'};
}

1;