#!/usr/bin/perl

undef $/;
$old = $ARGV[0];
$new = $ARGV[1];
foreach $name (@ARGV[ 2 .. $#ARGV ]) {
    rename $name, "$name.old";
    open(FILE,"$name.old");
    unlink "$name";
    open(NF, "> $name\0") or die "can't open output file!!";
    $line = <FILE>;
    $count = ($line =~ s/$old/$new/gm);
    print "replacing $count words in $name. \n";
    print NF $line;
    close FILE;
    unlink "$name.old";
};
