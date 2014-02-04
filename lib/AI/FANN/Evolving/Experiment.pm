package AI::FANN::Evolving::Experiment;
use strict;
use warnings;
use Algorithm::Genetic::Diploid;
use AI::FANN ':all';
use File::Temp 'tempfile';
use base 'Algorithm::Genetic::Diploid::Experiment';

my $log = __PACKAGE__->logger;

# get/set workdir
sub workdir {
	my $self = shift;
	if ( @_ ) {
		$log->info("assigning new workdir: @_");
		$self->{'workdir'} = shift;
	}
	return $self->{'workdir'};
}

# get/set TrainData object
sub traindata {
	my $self = shift;
	if ( @_ ) {
		$log->info("assigning new traindata: @_");
		$self->{'traindata'} = shift;
	}
	return $self->{'traindata'};
}

# the optimum is zero wrong predictions
sub optimum { 0 }

1;