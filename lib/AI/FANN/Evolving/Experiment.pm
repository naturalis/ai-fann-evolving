package AI::FANN::Evolving::Experiment;
use strict;
use warnings;
use Algorithm::Genetic::Diploid;
use AI::FANN ':all';
use File::Temp 'tempfile';
use base 'Algorithm::Genetic::Diploid::Experiment';

my $log = __PACKAGE__->logger;

=head1 NAME

AI::FANN::Evolving::Experiment - an experiment in evolving artificial intelligence

=head1 METHODS

=over

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

=item optimum

The optimal fitness is zero error in the ANN's classification. This method returns 
that value: 0.

=back

=cut

sub optimum { 0 }

1;