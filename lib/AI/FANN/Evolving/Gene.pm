package AI::FANN::Evolving::Gene;
use strict;
use warnings;
use List::Util 'shuffle';
use File::Temp 'tempfile';
use Scalar::Util 'refaddr';
use AI::FANN::Evolving;
use Algorithm::Genetic::Diploid::Gene;
use base 'Algorithm::Genetic::Diploid::Gene';
use Data::Dumper;

my $log = __PACKAGE__->logger;

=head1 NAME

AI::FANN::Evolving::Gene - gene that codes for an artificial neural network (ANN)

=head1 METHODS

=over

=item new

Constructor is passed named arguments. Instantiates a trained L<AI::FANN::Evolving> ANN

=cut

sub new {

	# initialize self up the inheritance tree
	my $self = shift->SUPER::new(@_);
			
	# instantiate and train the FANN object
	my $traindata = $self->experiment->traindata;
	$self->ann( AI::FANN::Evolving->new( 'data' => $traindata ) );
	return $self;
}

=item ann

Getter/setter for an L<AI::FANN::Evolving> ANN

=cut

sub ann {
	my $self = shift;
	if ( @_ ) {
		$log->debug("assigning and serializing ANN");
		my $ann = shift;
		my ( $fh, $ann_file ) = tempfile( 'DIR' => $self->experiment->workdir );
		$ann->save($ann_file);
		$self->{'ann'} = $ann_file;
		return $ann;
	}
	else {
		$log->debug("retrieving ANN");
	}
	if ( -e $self->{'ann'} ) {
		$log->debug("instantiating ANN from file");
		return AI::FANN::Evolving->new( 'file' => $self->{'ann'} );
	}
	else {
		$log->warn("ANN file doesn't exist");
	}
}

=item make_function

Returns a code reference to the fitness function, which when executed returns a fitness
value and writes the corresponding ANN to file

=cut

sub make_function {
	my $self = shift;
	my $ann = $self->ann;
	$log->debug("making fitness function");
	
	# build the fitness function
	return sub {		
	
		# isa TrainingData object, this is what we need to use
		# to make our prognostications. It is a different data 
		# set (out of sample) than the TrainingData object that
		# the AI was trained on.
		my $env = shift;		
		
		# this is a number which we try to keep as near to zero
		# as possible
		my $fitness = 0;
		
		# iterate over the list of input/output pairs
		for my $i ( 0 .. ( $env->length - 1 ) ) {
			my ( $input, $expected ) = $env->data($i);
			my $observed = $ann->run($input);
			
			# iterate over the observed and expected values
			for my $j ( 0 .. $#{ $expected } ) {
				$fitness += abs( $observed->[$j] - $expected->[$j] );	
			}
		}
		
		# store result
		$self->{'fitness'} = $fitness;

		# store the AI		
		my $outfile = $self->experiment->workdir . "/${fitness}.ann";
		$self->ann->save($outfile);
		return $self->{'fitness'};
	}
}

=item fitness

Stores the fitness value after expressing the fitness function

=cut

sub fitness { shift->{'fitness'} }

=item mutate

Mutates the ANN by stochastically altering its properties in proportion to 
the mutation_rate

=back

=cut

sub mutate {
	my $self = shift;
	
	# probably 0.05
	my $mu = $self->experiment->mutation_rate;

	# make a clone, which we might mutate further
	my $ann = $self->ann;
	my $ann_clone  = $ann->clone;
	my $self_clone = $self->clone;
	$log->debug("cloned ANN $ann => $ann_clone");
	
	# properties of ann we might mutate
	# XXX equally do this for discrete properties?
	for my $prop ( AI::FANN::Evolving->continuous_properties ) {
	
		# mutate by a value <= $mu
		my $change = rand($mu);
		my $propval  = $ann_clone->$prop;
		$propval = $mu if $propval == 0;
		my $npropval = $propval + ( $propval * $change - $propval * $mu / 2 );
		
		# we don't want the sign to change, e.g. a change from 0.01 to -0.01 might
		# have unforeseen effects, so flip the sign if we cross zero.
		$npropval *= -1 if $propval < 0 xor $npropval < 0;
		$log->debug("going to mutate $prop $propval => $npropval");
		$ann_clone->$prop( $npropval );
	}
	my $data = $self->experiment->traindata;
	$log->debug("going to re-train using $data");
	$ann_clone->train( $self->experiment->traindata );
	$self_clone->ann( $ann_clone );
	return $self_clone;
}

1;
