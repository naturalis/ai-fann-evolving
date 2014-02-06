package AI::FANN::Evolving::TrainData;
use strict;
use Algorithm::Genetic::Diploid::Logger;

my $log = Algorithm::Genetic::Diploid::Logger->new;

sub new {
	my $class = shift;
	my %args  = @_;
	my $self  = {
		'skip'      => $args{'skip'}      || [],
		'dependent' => $args{'dependent'} || undef,
		'offset'    => $args{'offset'}    || 1,
		'header'    => {},
		'table'     => [],
		'size'      => 0,
	};
	bless $self, $class;
	$self->read_data($args{'file'}) if $args{'file'};
	$self->trim_data if $args{'trim'};
	return $self;
}

sub skip {
	my $self = shift;
	$self->{'skip'} = \@_ if @_;
	return @{ $self->{'skip'} };
}

sub get_predictor_columns {
	my $self = shift;
	my %skip = map { $_ => 1 } $self->skip;
	return grep { ! $skip{$_} } keys %{ $self->{'header'} };
}

sub to_fann {
	my ( $self, @cols ) = @_;
	my @deps = $self->get_dependent;
	my @pred = $self->get_predictors( 'cols' => \@cols );
	my @interdigitated;
	my $ncols;
	for my $i ( 0 .. $#deps ) {
		$ncols = scalar @{ $pred[$i] };
		push @interdigitated, $pred[$i], [ $deps[$i] ];
	}
	$log->debug("number of columns: $ncols");
	return AI::FANN::TrainData->new(@interdigitated);
}

sub read_data {
	my ( $self, $file ) = @_; # file is tab-delimited
	open my $fh, '<', $file or die "Can't open $file: $!";
	my ( %header, @table );
	while(<$fh>) {
		chomp;
		my @fields = split /\t/, $_;
		if ( not %header ) {
			my $i = 0;
			%header = map { $_ => $i++ } @fields;
		}
		else {
			push @table, \@fields;
		}
	}
	$self->{'header'} = \%header;
	$self->{'table'}  = \@table;
	$self->{'size'}   = scalar @table;
	return $self;
}

sub write_data {
	my ( $self, $file ) = @_;
	
	# use file or STDOUT
	my $fh;
	if ( $file ) {
		open $fh, '>', $file or die "Can't write to $file: $!";
		$log->info("writing data to $file");
	}
	else {
		$fh = \*STDOUT;
		$log->info("writing data to STDOUT");
	}
	
	# print header
	my $h = $self->{'header'};
	print $fh join "\t", sort { $h->{$a} <=> $h->{$b} } keys %{ $h };
	print $fh "\n";
	
	# print rows
	for my $row ( @{ $self->{'table'} } ) {
		print $fh join "\t", @{ $row };
		print $fh "\n";
	}
}

sub trim_data {
	my $self = shift;
	my @trimmed;
	ROW: for my $row ( @{ $self->{'table'} } ) {
		next ROW if grep { not defined $_ } @{ $row };
		push @trimmed, $row;
	}
	my $num = $self->{'size'} - scalar @trimmed;
	$log->info("removed $num incomplete rows");
	$self->{'size'}  = scalar @trimmed;
	$self->{'table'} = \@trimmed;
}

sub get_predictor_count {
	my $self = shift;
	my %skip = map { $_ => 1 } @{ $self->{'skip'} };
	my @h = sort { $a cmp $b } grep { ! $skip{$_} } keys %{ $self->{'header'} };
	return scalar @h;
}

sub get_predictors {
	my ( $self, %args ) = @_;
	my $i = $args{'row'};
	my @cols = @{ $args{'cols'} } if $args{'cols'};
	
	# build hash of indices to skip
	my $dep = $self->{'dependent'};
	my $dep_idx = $self->{'header'}->{$dep};
	my %skip = map { $self->{'header'}->{$_} => 1 } @{ $self->{'skip'} };
	$skip{$dep_idx} = 1;
	
	# build hash of indices to keep
	my %keep = map { $self->{'header'}->{$_} => 1 } @cols;
	
	# only return a single row
	if ( defined $i ) {
		my @pred;
		for my $j ( 0 .. $#{ $self->{'table'}->[$i] } ) {
			if ( %keep ) {
				push @pred, $self->{'table'}->[$i]->[$j] if $keep{$j};
			}
			else {
				push @pred, $self->{'table'}->[$i]->[$j] unless $skip{$j};
			}
		}
		return @pred;
	}
	else {
		my @preds;
		my $max = $self->{'size'} - 1;
		for my $j ( 0 .. $max ) {
			push @preds, [ $self->get_predictors( 'row' => $j, 'cols' => \@cols) ];
		}
		return @preds;
	}
}

sub get_dependent {
	my ( $self, $i ) = @_;
	if ( defined $i ) {
		my $dc  = $self->{'header'}->{ $self->{'dependent'} };
		my $off = $self->{'offset'};
		return $self->{'table'}->[ $i + $off ]->[ $dc ];
	}
	else {
		my $dc   = $self->{'header'}->{ $self->{'dependent'} };
		my $off  = $self->{'offset'};
		my $max  = $self->{'size'} - ( 1 + $off );
		my @dep  = map { $self->{'table'}->[ $_ + $off ]->[ $dc ] } 0 .. $max;
		return @dep;
	}
}

1;
