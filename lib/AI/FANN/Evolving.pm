package AI::FANN::Evolving;
use strict;
use warnings;
use AI::FANN ':all';
use File::Temp 'tempfile';
use Scalar::Util 'refaddr';
use AI::FANN::Evolving::Gene;
use AI::FANN::Evolving::Chromosome;
use AI::FANN::Evolving::Experiment;
use AI::FANN::Evolving::Factory;
use Algorithm::Genetic::Diploid;
use Algorithm::Genetic::Diploid::Logger;
use base 'AI::FANN';

our $VERSION = '0.3';
my $log = Algorithm::Genetic::Diploid::Logger->new;
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
		$log->debug("instantiating from file $file");
		return $self->_init(%args);
	}
	
	# build new topology
	elsif ( my $data = $args{'data'} ) {
		$log->debug("instantiating from data $data");
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

my %default = (
	'error'               => 0.0001,
	'epochs'              => 5000,
	'train_type'          => 'ordinary',
	'epoch_printfreq'     => 100,
	'neuron_printfreq'    => 0,
	'neurons'             => 15,
	'activation_function' => FANN_SIGMOID_SYMMETRIC,
);

my %constant = (
	# enum fann_train_enum
	'FANN_TRAIN_INCREMENTAL' => FANN_TRAIN_INCREMENTAL,
	'FANN_TRAIN_BATCH'       => FANN_TRAIN_BATCH,
	'FANN_TRAIN_RPROP'       => FANN_TRAIN_RPROP,
	'FANN_TRAIN_QUICKPROP'   => FANN_TRAIN_QUICKPROP,

	# enum fann_activationfunc_enum
	'FANN_LINEAR'                     => FANN_LINEAR,
	'FANN_THRESHOLD'                  => FANN_THRESHOLD,
	'FANN_THRESHOLD_SYMMETRIC'        => FANN_THRESHOLD_SYMMETRIC,
	'FANN_SIGMOID'                    => FANN_SIGMOID,
	'FANN_SIGMOID_STEPWISE'           => FANN_SIGMOID_STEPWISE,
	'FANN_SIGMOID_SYMMETRIC'          => FANN_SIGMOID_SYMMETRIC,
	'FANN_SIGMOID_SYMMETRIC_STEPWISE' => FANN_SIGMOID_SYMMETRIC_STEPWISE,
	'FANN_GAUSSIAN'                   => FANN_GAUSSIAN,
	'FANN_GAUSSIAN_SYMMETRIC'         => FANN_GAUSSIAN_SYMMETRIC,
	'FANN_GAUSSIAN_STEPWISE'          => FANN_GAUSSIAN_STEPWISE,
	'FANN_ELLIOT'                     => FANN_ELLIOT,
	'FANN_ELLIOT_SYMMETRIC'           => FANN_ELLIOT_SYMMETRIC,
	'FANN_LINEAR_PIECE'               => FANN_LINEAR_PIECE,
	'FANN_LINEAR_PIECE_SYMMETRIC'     => FANN_LINEAR_PIECE_SYMMETRIC,
	'FANN_SIN_SYMMETRIC'              => FANN_SIN_SYMMETRIC,
	'FANN_COS_SYMMETRIC'              => FANN_COS_SYMMETRIC,
	'FANN_SIN'                        => FANN_SIN,
	'FANN_COS'                        => FANN_COS,

	# enum fann_errorfunc_enum
	'FANN_ERRORFUNC_LINEAR' => FANN_ERRORFUNC_LINEAR,
	'FANN_ERRORFUNC_TANH'   => FANN_ERRORFUNC_TANH,

	# enum fann_stopfunc_enum
	'FANN_STOPFUNC_MSE' => FANN_STOPFUNC_MSE,
	'FANN_STOPFUNC_BIT' => FANN_STOPFUNC_BIT,
);

sub defaults {
	my $self = shift;
	my %args = @_;
	for my $key ( keys %args ) {
		$log->info("setting $key to $args{$key}");
		if ( $key eq 'activation_function' ) {
			$args{$key} = $constant{$args{$key}};
		}
		$default{$key} = $args{$key};
	}
}

sub _init {
	my $self = shift;
	my $id   = refaddr $self;
	my %args = @_;
	$self{$id} = {};
	for ( qw(error epochs train_type epoch_printfreq neuron_printfreq neurons activation_function) ) {
		$self{$id}->{$_} = $args{$_} // $default{$_};
	}
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
	$log->debug("cloning...");
	my ( $fh, $name ) = tempfile();
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
		$log->debug("cascade training");
	
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
		$log->debug("normal training");
	
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
		'learning_momentum',
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
	(
	
	
	);
}

=item error

Getter/setter for the error rate. Default is 0.0001

=cut

sub error {
	my $ref = shift;
	my $id = refaddr $ref;
	if ( @_ ) {
		my $value = shift;
		$log->debug("setting error threshold to $value");
		$self{$id}->{'error'} = $value;
	}
	else {
		$log->debug("retrieving error threshold");
	}
	return $self{$id}->{'error'};
}

=item epochs

Getter/setter for the number of training epochs, default is 500000

=cut

sub epochs {
	my $ref = shift;
	my $id = refaddr $ref;
	if ( @_ ) {
		my $value = shift;
		$log->debug("setting training epochs to $value");
		$self{$id}->{'epochs'} = $value;
	}
	else {
		$log->debug("retrieving training epochs");
	}
	return $self{$id}->{'epochs'};
}

=item epoch_printfreq

Getter/setter for the number of epochs after which progress is printed. default is 1000

=cut

sub epoch_printfreq {
	my $ref = shift;
	my $id = refaddr $ref;
	if ( @_ ) {
		my $value = shift;
		$log->debug("setting epoch printfreq to $value");
		$self{$id}->{'epoch_printfreq'} = $value;
	}
	else {
		$log->debug("retrieving epoch printfreq");
	}
	return $self{$id}->{'epoch_printfreq'};
}

=item neurons

Getter/setter for the number of neurons. Default is 15

=cut

sub neurons {
	my $ref = shift;
	my $id = refaddr $ref;
	if ( @_ ) {
		my $value = shift;
		$log->debug("setting neurons to $value");
		$self{$id}->{'neurons'} = $value;
	}
	else {
		$log->debug("retrieving neurons");
	}
	return $self{$id}->{'neurons'};
}

=item neuron_printfreq

Getter/setter for the number of cascading neurons after which progress is printed. 
default is 10

=cut

sub neuron_printfreq {
	my $ref = shift;
	my $id = refaddr $ref;
	if ( @_ ) {
		my $value = shift;
		$log->debug("setting neuron printfreq to $value");
		$self{$id}->{'neuron_printfreq'} = $value;
	}
	else {	
		$log->debug("retrieving neuron printfreq");
	}
	return $self{$id}->{'neuron_printfreq'};
}

=item train_type

Getter/setter for the training type: 'cascade' or 'ordinary'. Default is ordinary

=cut

sub train_type {
	my $ref = shift;
	my $id = refaddr $ref;
	if ( @_ ) {
		my $value = lc shift;
		$log->debug("setting train type to $value"); 
		$self{$id}->{'train_type'} = $value;
	}
	else {
		$log->debug("retrieving train type");
	}
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
	if ( @_ ) {
		my $value = shift;
		$log->debug("setting activation function to $value");
		$self{$id}->{'activation_function'} = $value;
	}
	else {
		$log->debug("retrieving activation function");
	}
	return $self{$id}->{'activation_function'};
}

1;
