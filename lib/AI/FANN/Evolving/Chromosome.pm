package AI::FANN::Evolving::Chromosome;
use strict;
use AI::FANN::Evolving;
use AI::FANN::Evolving::Experiment;
use Algorithm::Genetic::Diploid;
use base 'Algorithm::Genetic::Diploid::Chromosome';

my $log = __PACKAGE__->logger;

=head1 NAME

AI::FANN::Evolving::Chromosome - chromosome of an evolving, diploid AI

=head1 METHODS

=over

=item recombine

Recombines properties of the AI during meiosis in proportion to the crossover_rate

=back

=cut

sub recombine {
	$log->debug("recombining chromosomes");
	# get the genes and columns for the two chromosomes
	my ( $chr1, $chr2 ) = @_;
	my ( $gen1 ) = map { $_->mutate } $chr1->genes;
	my ( $gen2 ) = map { $_->mutate } $chr2->genes;	
	my ( $ann1, $ann2 ) = ( $gen1->ann, $gen2->ann );
	my $exp = $chr1->experiment;
	
	# XXX equally do this for discrete properties?
	for my $prop ( AI::FANN::Evolving->continuous_properties ) {
	
		# switch values every time rand() is below crossover
		my $rate = $exp->crossover_rate;
		my $val = rand(1);
		if ( $val <= $rate ) {
			$log->debug("recombination of $prop");
			my $prop1 = $ann1->$prop;
			my $prop2 = $ann2->$prop;
			$ann1->$prop($prop2);
			$ann2->$prop($prop1);
		}
	}
	
	# assign the genes to the chromosomes (this because they are clones
	# so we can't use the old object reference)
	$chr1->genes($gen1);
	$chr2->genes($gen2);	
}

1;
