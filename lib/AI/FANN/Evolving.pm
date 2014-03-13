package AI::FANN::Evolving;
use strict;
use warnings;
use AI::FANN ':all';
use File::Temp 'tempfile';
use AI::FANN::Evolving::Gene;
use AI::FANN::Evolving::Chromosome;
use AI::FANN::Evolving::Experiment;
use AI::FANN::Evolving::Factory;
use Algorithm::Genetic::Diploid;
use base qw'Algorithm::Genetic::Diploid::Base';

our $VERSION = '0.3';
our $AUTOLOAD;
my $log = __PACKAGE__->logger;

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
	my $self  = {};
	bless $self, $class;
	$self->_init(%args);
	
	# de-serialize from a file
	if ( my $file = $args{'file'} ) {
		$self->{'ann'} = AI::FANN->new_from_file($file);
		$log->debug("instantiating from file $file");
		return $self;
	}
	
	# build new topology from input data
	elsif ( my $data = $args{'data'} ) {
		$log->debug("instantiating from data $data");
		$data = $data->to_fann if $data->isa('AI::FANN::Evolving::TrainData');
		
		# prepare arguments
		my $neurons = $args{'neurons'} || ( $data->num_inputs + 1 );
		my @sizes = ( 
			$data->num_inputs, 
			$neurons,
			$data->num_outputs
		);
		
		# build topology
		if ( $args{'connection_rate'} ) {
			$self->{'ann'} = AI::FANN->new_sparse( $args{'connection_rate'}, @sizes );
		}
		else {
			$self->{'ann'} = AI::FANN->new_standard( @sizes );
		}
		
		# finalize the instance
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
	my %args = @_;
	for ( qw(error epochs train_type epoch_printfreq neuron_printfreq neurons activation_function) ) {
		$self->{$_} = $args{$_} // $default{$_};
	}
	return $self;
}

=item clone

Clones the object

=cut

sub clone {
	my $self = shift;
	$log->debug("cloning...");
	
	# we delete the reference here so we can use 
	# Algorithm::Genetic::Diploid::Base's cloning method, which
	# dumps and loads from YAML. This wouldn't work if the 
	# reference is still attached because it cannot be 
	# stringified, being an XS data structure
	my $ann = delete $self->{'ann'};
	my $clone = $self->SUPER::clone;
	
	# clone the ANN by writing it to a temp file in "FANN/FLO"
	# format and reading that back in
	my ( $fh, $file ) = tempfile();
	close $fh;
	$ann->save($file);
	$clone->{'ann'} = __PACKAGE__->new_from_file($file);
	unlink $file;
	
	# now re-attach the original ANN to the invocant
	$self->{'ann'} = $ann;
	
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
		$self->{'ann'}->cascadetrain_on_data(
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
		$self->{'ann'}->train_on_data(
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

=item enum_properties

Returns a hash whose keys are names of enums and values the possible states for the
enum

=cut

sub enum_properties {

}

=item error

Getter/setter for the error rate. Default is 0.0001

=cut

sub error {
	my $self = shift;
	if ( @_ ) {
		my $value = shift;
		$log->debug("setting error threshold to $value");
		return $self->{'error'} = $value;
	}
	else {
		$log->debug("getting error threshold");
		return $self->{'error'};
	}
}

=item epochs

Getter/setter for the number of training epochs, default is 500000

=cut

sub epochs {
	my $self = shift;
	if ( @_ ) {
		my $value = shift;
		$log->debug("setting training epochs to $value");
		return $self->{'epochs'} = $value;
	}
	else {
		$log->debug("getting training epochs");
		return $self->{'epochs'};
	}
}

=item epoch_printfreq

Getter/setter for the number of epochs after which progress is printed. default is 1000

=cut

sub epoch_printfreq {
	my $self = shift;
	if ( @_ ) {
		my $value = shift;
		$log->debug("setting epoch printfreq to $value");
		return $self->{'epoch_printfreq'} = $value;
	}
	else {
		$log->debug("getting epoch printfreq");
		return $self->{'epoch_printfreq'}
	}
}

=item neurons

Getter/setter for the number of neurons. Default is 15

=cut

sub neurons {
	my $self = shift;
	if ( @_ ) {
		my $value = shift;
		$log->debug("setting neurons to $value");
		return $self->{'neurons'} = $value;
	}
	else {
		$log->debug("getting neurons");
		return $self->{'neurons'};
	}
}

=item neuron_printfreq

Getter/setter for the number of cascading neurons after which progress is printed. 
default is 10

=cut

sub neuron_printfreq {
	my $self = shift;
	if ( @_ ) {
		my $value = shift;
		$log->debug("setting neuron printfreq to $value");
		return $self->{'neuron_printfreq'} = $value;
	}
	else {	
		$log->debug("getting neuron printfreq");
		return $self->{'neuron_printfreq'};
	}
}

=item train_type

Getter/setter for the training type: 'cascade' or 'ordinary'. Default is ordinary

=cut

sub train_type {
	my $self = shift;
	if ( @_ ) {
		my $value = lc shift;
		$log->debug("setting train type to $value"); 
		return $self->{'train_type'} = $value;
	}
	else {
		$log->debug("getting train type");
		return $self->{'train_type'};
	}
}

=item activation_function

Getter/setter for the function that maps inputs to outputs. default is 
FANN_SIGMOID_SYMMETRIC

=back

=cut

sub activation_function {
	my $self = shift;
	if ( @_ ) {
		my $value = shift;
		$log->debug("setting activation function to $value");
		return $self->{'activation_function'} = $value;
	}
	else {
		$log->debug("getting activation function");
		return $self->{'activation_function'};
	}
}

# this is here so that we can trap method calls that need to be 
# delegated to the FANN object. at this point we're not even
# going to care whether the FANN object implements these methods:
# if it doesn't we get the normal error for unknown methods, which
# the user then will have to resolve.
sub AUTOLOAD {
	my $self = shift;
	my $method = $AUTOLOAD;
	$method =~ s/.+://;
	
	# ignore all caps methods
	if ( $method !~ /^[A-Z]+$/ ) {
		my $ann = $self->{'ann'};
		if ( @_ ) {
			return $ann->$method(shift);
		}
		else {
			return $ann->$method;
		}
	}
	
}

1;
