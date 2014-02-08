package AI::FANN::Evolving::Experiment;
use strict;
use warnings;
use AI::FANN ':all';
use AI::FANN::Evolving;
use File::Temp 'tempfile';
use Algorithm::Genetic::Diploid;
use base 'Algorithm::Genetic::Diploid::Experiment';

my $log = __PACKAGE__->logger;

=head1 NAME

AI::FANN::Evolving::Experiment - an experiment in evolving artificial intelligence

=head1 METHODS

=over

=item new

Constructor takes named arguments, sets default factory to L<AI::FANN::Evolving::Factory>

=cut

sub new { shift->SUPER::new( 'factory' => AI::FANN::Evolving::Factory->new, @_ ) }

=item workdir

Getter/Setter for the workdir where L<AI::FANN> artificial neural networks will be
written during the experiment. The files will be named after the ANN's error, which 
needs to be minimized.

=cut

sub workdir {
	my $self = shift;
	if ( @_ ) {
		$log->info("assigning new workdir: @_");
		$self->{'workdir'} = shift;
	}
	return $self->{'workdir'};
}

=item traindata

Getter/setter for the L<AI::FANN::TrainData> object.

=cut

sub traindata {
	my $self = shift;
	if ( @_ ) {
		$log->info("assigning new traindata: @_");
		$self->{'traindata'} = shift;
	}
	return $self->{'traindata'};
}

=item run

Runs the experiment!

=cut

sub run {
	my $self = shift;
	my $log = $self->logger;
	
	$log->info("going to run experiment");
	my @results;
	for my $i ( 1 .. $self->ngens ) {
	
		# modify workdir
		my $wd = $self->{'workdir'};
		$wd =~ s/\d+$/$i/;
		$self->{'workdir'} = $wd;
		mkdir $wd;
		
		my $optimum = $self->optimum($i);
		
		$log->debug("optimum at generation $i is $optimum");
		my ( $fittest, $fitness ) = $self->population->turnover($i,$self->env,$optimum);
		push @results, [ $fittest, $fitness ];
	}
	my ( $fittest, $fitness ) = map { @{ $_ } } sort { $a->[1] <=> $b->[1] } @results;
	return $fittest, $fitness;
}

=item optimum

The optimal fitness is zero error in the ANN's classification. This method returns 
that value: 0.

=back

=cut

sub optimum { 0 }

1;
